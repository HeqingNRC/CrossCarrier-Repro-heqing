"""Checkpoint-selection diagnostic.

Goal: with ONE seed, find which TARGET-FREE checkpoint-selection rule (source-val
metric / robustness / invariance / plateau / weight-soup / hybrid) best
approximates the true 77GHz target-best ("oracle"). The paper cannot say "we
selected on the target"; this tells us which fixed, interpretable, source-only
rule loses the least vs that oracle (regret) and which source signal actually
tracks 77GHz F1 (rank correlation).

Inputs:
  run_dir/epoch_ckpts/ep###.pt   -- per-epoch TRAINABLE-only weights {raw, ema}
                                     (dumped by the instrumented trainer)
Eval sets (this script does ALL target evaluation itself):
  source-val  = val.csv   (10/24 GHz, 162 imgs)  -> every source-only signal
  target      = test.csv  (77 GHz, full 418, val merged into test)  -> ORACLE

Everything that does not depend on the weights (clean/augmented/invariance image
tensors) is precomputed ONCE and cached on GPU; only the per-epoch trainable
weights change, so each epoch is just a few forward passes.

Output:
  reports_diag/diagnostic.json
  reports_diag/diagnostic.md     (ranked regret table + correlation table)
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

import config
import v9_2_1lib as lib

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = torch.bfloat16
CLASSES = config.CLASSES
NC = config.NUM_CLASSES
ROOT = config.DATASET_ROOT
EMA_START = int(getattr(config, "EMA_START_EPOCH", 5))

# Fixed, deterministic robustness/invariance probe carriers (GHz). The method is
# about cross-carrier DAS generalization, so the natural source-only stress test
# is to DAS the source-val toward higher (deployment/OOD) carriers and measure
# how stable the model stays. These are FIXED so the rule is reproducible.
AUG_CARRIERS = [50.0, 77.0, 99.0, 120.0]   # robustness bank
INV_CARRIER = 77.0                          # invariance reference (deployment)


# --------------------------------------------------------------------------- #
# Image cache (deterministic eval transform; matches training input exactly).  #
# --------------------------------------------------------------------------- #
_eval_tf = lib.tensor_transform(train=False)


def _load_tensor(path, f_src=None, f_virt=None):
    img = Image.open(ROOT / path).convert("RGB")
    if f_virt is not None and f_src is not None and abs(f_virt - f_src) > 1e-9:
        img = lib.das_deterministic(img, f_src, f_virt)
    return _eval_tf(img)


def build_cache(df, carriers=None):
    """Return dict: 'clean' -> (N,3,H,W) GPU tensor, and per-carrier DAS views."""
    paths = df["path"].tolist()
    fsrc = [lib.parse_freq_ghz(f) for f in df["frequency"].tolist()]
    out = {}
    clean = torch.stack([_load_tensor(p) for p in paths]).to(DEVICE)
    out["clean"] = clean
    if carriers:
        for fv in carriers:
            out[fv] = torch.stack(
                [_load_tensor(p, fs, fv) for p, fs in zip(paths, fsrc)]
            ).to(DEVICE)
    return out


# --------------------------------------------------------------------------- #
# Forward / metrics                                                            #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def forward_logits(model, x, bs=256):
    outs = []
    for i in range(0, x.shape[0], bs):
        with torch.autocast("cuda", dtype=AMP):
            z = model.encode_adapted(x[i:i + bs])          # skip frozen oracle
            lg = model.logits_from_neck(z, margin=False)   # ArcFace-scaled cosine
        outs.append(lg.float().cpu())
    return torch.cat(outs, 0)


def _f1_macro(preds, y):
    f1s = []
    for c in range(NC):
        tp = int(((preds == c) & (y == c)).sum())
        fp = int(((preds == c) & (y != c)).sum())
        fn = int(((preds != c) & (y == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(f1s)), f1s


def _ece(conf, correct, nbins=15):
    bins = np.linspace(0, 1, nbins + 1)
    e = 0.0
    n = len(conf)
    for i in range(nbins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.any():
            e += abs(conf[m].mean() - correct[m].mean()) * m.sum() / n
    return float(e)


def metric_panel(logits, y, groups=None):
    """Full metric panel from logits (T tensor) + labels (np)."""
    p = torch.softmax(logits, 1).numpy()
    lg = logits.numpy()
    preds = p.argmax(1)
    y = np.asarray(y)
    onehot = np.eye(NC)[y]
    acc = float((preds == y).mean())
    macro_f1, f1s = _f1_macro(preds, y)
    recalls = [float((preds[y == c] == c).mean()) if (y == c).any() else 0.0
               for c in range(NC)]
    bal_acc = float(np.mean(recalls))
    worst_class_f1 = float(np.min(f1s))
    nll = float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1)).mean())
    brier = float(((p - onehot) ** 2).sum(1).mean())
    conf = p.max(1)
    correct = (preds == y).astype(float)
    ece = _ece(conf, correct)
    # margin: correct-class logit minus best other-class logit
    corr_logit = lg[np.arange(len(y)), y]
    tmp = lg.copy(); tmp[np.arange(len(y)), y] = -1e9
    margin = float((corr_logit - tmp.max(1)).mean())
    entropy = float((-(p * np.log(np.clip(p, 1e-9, 1))).sum(1)).mean())
    hi_conf_err = float(((conf > 0.9) & (correct == 0)).mean())
    out = {
        "acc": acc, "macro_f1": macro_f1, "bal_acc": bal_acc,
        "worst_class_f1": worst_class_f1, "nll": nll, "brier": brier,
        "ece": ece, "margin": margin, "entropy": entropy,
        "hi_conf_err": hi_conf_err,
    }
    if groups is not None:
        gf1 = []
        for g in sorted(set(groups)):
            m = np.asarray(groups) == g
            if m.sum() >= NC:  # need enough samples for a meaningful macro-F1
                gf1.append(_f1_macro(preds[m], y[m])[0])
        out["worst_group_f1"] = float(np.min(gf1)) if gf1 else macro_f1
    return out, p


# --------------------------------------------------------------------------- #
# Model build + per-epoch overlay                                             #
# --------------------------------------------------------------------------- #
def build_model(backbone, adapter_mode):
    m = lib.TimmBackboneV921(backbone, NC, adapter_mode=adapter_mode).to(DEVICE)
    m.eval()
    return m


def overlay(model, state):
    model.load_state_dict(state, strict=False)


# --------------------------------------------------------------------------- #
# Main sweep                                                                   #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--basis", default="ema", choices=["ema", "raw"])
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    ck_dir = run_dir / "epoch_ckpts"
    out_dir = run_dir / "reports_diag"
    out_dir.mkdir(parents=True, exist_ok=True)

    ck_files = sorted(ck_dir.glob("ep*.pt"))
    assert ck_files, f"no epoch checkpoints in {ck_dir}"
    meta0 = torch.load(ck_files[0], map_location="cpu")
    backbone, adapter_mode = meta0["backbone"], meta0["adapter_mode"]
    print(f"[diag] {len(ck_files)} epochs  backbone={backbone}  basis={args.basis}")

    # ---- data ----
    val_df = lib.load_manifest("val", keep_7c=True).reset_index(drop=True)
    tgt_df = lib.load_manifest("test", keep_7c=True).reset_index(drop=True)
    yv = val_df["class_idx_7c"].to_numpy()
    yt = tgt_df["class_idx_7c"].to_numpy()
    # groups for worst-group F1 on source-val: subject x frequency
    vgroups = (val_df["subject"].astype(str) + "|" + val_df["frequency"].astype(str)).tolist()
    print(f"[diag] source-val={len(val_df)}  target77={len(tgt_df)} (full, val-merged-into-test)")

    print("[diag] building image caches (once)...")
    val_cache = build_cache(val_df, carriers=AUG_CARRIERS)   # clean + 4 DAS views
    tgt_cache = build_cache(tgt_df, carriers=None)           # clean only (oracle)
    print("[diag] caches ready.")

    model = build_model(backbone, adapter_mode)

    # ---- per-epoch metric panel ----
    rows = []
    tgt_probs_by_ep = {}      # for last-k prediction ensemble (EMA)
    states_by_ep = {}         # EMA trainable states (primary basis for panel)
    states_raw_by_ep = {}     # raw/live trainable states (for raw fixed rules)
    for f in ck_files:
        ck = torch.load(f, map_location="cpu")
        ep = int(ck["epoch"])
        state = ck[args.basis]
        states_by_ep[ep] = state
        states_raw_by_ep[ep] = ck["raw"]
        overlay(model, {k: v.to(DEVICE) for k, v in state.items()})

        # source-val clean panel
        lv = forward_logits(model, val_cache["clean"])
        src, pv_clean = metric_panel(lv, yv, groups=vgroups)

        # robustness bank (DAS source-val to higher carriers)
        aug_f1s, aug_ents, consist = [], [], []
        pv_aug_avg = np.zeros_like(pv_clean)
        for fv in AUG_CARRIERS:
            la = forward_logits(model, val_cache[fv])
            pa = torch.softmax(la, 1).numpy()
            aug_f1s.append(_f1_macro(pa.argmax(1), yv)[0])
            aug_ents.append(float((-(pa * np.log(np.clip(pa, 1e-9, 1))).sum(1)).mean()))
            consist.append(float((pa.argmax(1) == pv_clean.argmax(1)).mean()))
            pv_aug_avg += pa / len(AUG_CARRIERS)
        aug_f1_mean = float(np.mean(aug_f1s))
        worst_aug_f1 = float(np.min(aug_f1s))
        aug_consistency = float(np.mean(consist))
        aug_entropy = float(np.mean(aug_ents))

        # invariance: feature cosine + prediction KL between clean and DAS->77
        with torch.no_grad():
            zc = model.encode_adapted(val_cache["clean"].to(DEVICE)).float()
            zi = model.encode_adapted(val_cache[INV_CARRIER].to(DEVICE)).float()
        feat_cos = float(F.cosine_similarity(zc, zi, dim=1).mean().cpu())
        pc = torch.softmax(lv, 1)
        li = forward_logits(model, val_cache[INV_CARRIER])
        pi = torch.softmax(li, 1)
        pred_kl = float((pc * (torch.log(pc.clamp_min(1e-9)) -
                               torch.log(pi.clamp_min(1e-9)))).sum(1).mean())
        # feature-carrier separability (Fisher ratio of neck feats wrt 10 vs 24)
        fr = val_df["frequency"].to_numpy()
        zc_np = zc.cpu().numpy()
        m10, m24 = zc_np[fr == "10GHz"].mean(0), zc_np[fr == "24GHz"].mean(0)
        pooled = zc_np.std(0) + 1e-6
        carrier_fisher = float(np.mean(np.abs(m10 - m24) / pooled))

        # target oracle panel (full 418)
        lt = forward_logits(model, tgt_cache["clean"])
        tgt, pt = metric_panel(lt, yt)
        tgt_probs_by_ep[ep] = pt

        rows.append({
            "epoch": ep,
            # source-val signals (target-free)
            "src_acc": src["acc"], "src_f1": src["macro_f1"],
            "src_bal_acc": src["bal_acc"], "src_worst_class_f1": src["worst_class_f1"],
            "src_worst_group_f1": src["worst_group_f1"],
            "src_nll": src["nll"], "src_brier": src["brier"], "src_ece": src["ece"],
            "src_margin": src["margin"], "src_entropy": src["entropy"],
            "src_hi_conf_err": src["hi_conf_err"],
            "aug_f1_mean": aug_f1_mean, "worst_aug_f1": worst_aug_f1,
            "aug_consistency": aug_consistency, "aug_entropy": aug_entropy,
            "feat_cos_inv": feat_cos, "pred_kl_inv": pred_kl,
            "carrier_fisher": carrier_fisher,
            # oracle (target) -- NOT used by any rule except the oracle itself
            "tgt_f1": tgt["macro_f1"], "tgt_acc": tgt["acc"],
        })
        print(f"  ep{ep:03d}  src_f1={src['macro_f1']:.4f} src_acc={src['acc']:.4f} "
              f"aug_f1={aug_f1_mean:.4f} kl_inv={pred_kl:.4f}  ||  tgt_f1={tgt['macro_f1']:.4f}")

    hist = pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)
    hist.to_csv(out_dir / "per_epoch_metrics.csv", index=False)

    # ===================================================================== #
    # Selection rules                                                       #
    # ===================================================================== #
    eps = hist["epoch"].to_numpy()
    tgt_f1 = hist["tgt_f1"].to_numpy()
    valid = eps >= EMA_START                       # EMA basis meaningful only here
    H = hist[valid].reset_index(drop=True)
    oracle_idx = int(np.argmax(H["tgt_f1"].to_numpy()))
    oracle_ep = int(H["epoch"][oracle_idx]); oracle_f1 = float(H["tgt_f1"][oracle_idx])

    def tgt_at(ep):
        return float(hist.loc[hist["epoch"] == ep, "tgt_f1"].iloc[0])

    results = []   # (category, rule, picked, tgt_f1)

    def pick_by(col, maximize=True, label=None, cat="source-val"):
        s = H[col].to_numpy()
        i = int(np.argmax(s) if maximize else np.argmin(s))
        ep = int(H["epoch"][i])
        results.append((cat, label or col, f"ep{ep}", tgt_at(ep)))

    # ---- fixed rules ----
    last_ep = int(eps.max())

    def eval_target_state(state):
        overlay(model, {kk: v.to(DEVICE) for kk, v in state.items()})
        lt = forward_logits(model, tgt_cache["clean"])
        return _f1_macro(torch.softmax(lt, 1).numpy().argmax(1), yt)[0]

    def weight_soup(store, k):
        sel = sorted(store)[-k:]
        keys = store[sel[0]].keys()
        avg = {kk: sum(store[e][kk].float() for e in sel) / len(sel) for kk in keys}
        return eval_target_state(avg)

    results.append(("fixed", "final_ema", f"ep{last_ep}", tgt_at(last_ep)))
    results.append(("fixed", "final_checkpoint(raw)", f"ep{last_ep}",
                    eval_target_state(states_raw_by_ep[last_ep])))
    for k in (5, 10):
        results.append(("fixed", f"ema_weight_soup_last{k}", f"last{k}",
                        weight_soup(states_by_ep, k)))
        results.append(("fixed", f"raw_weight_soup_last{k}", f"last{k}",
                        weight_soup(states_raw_by_ep, k)))

    def pred_ensemble(k):
        sel = sorted(tgt_probs_by_ep)[-k:]
        pavg = sum(tgt_probs_by_ep[e] for e in sel) / len(sel)
        return _f1_macro(pavg.argmax(1), yt)[0]

    for k in (5, 10):
        results.append(("fixed", f"pred_ensemble_last{k}", f"last{k}", pred_ensemble(k)))

    # ---- source-val metric rules ----
    pick_by("src_acc", True, "best_src_acc")
    pick_by("src_f1", True, "best_src_macro_f1")
    pick_by("src_bal_acc", True, "best_src_bal_acc")
    pick_by("src_worst_class_f1", True, "best_src_worst_class_f1")
    pick_by("src_worst_group_f1", True, "best_src_worst_group_f1")
    pick_by("src_nll", False, "best_src_nll(low)")
    pick_by("src_brier", False, "best_src_brier(low)")
    pick_by("src_ece", False, "best_src_ece(low)")
    pick_by("src_margin", True, "best_src_margin")
    pick_by("src_hi_conf_err", False, "min_src_hi_conf_err")

    # ---- plateau rules (on source-val acc) ----
    def first_reach(thr):
        m = H[H["src_acc"] >= thr]
        return int(m["epoch"].iloc[0]) if len(m) else None
    for thr in (0.985, 0.99):
        ep = first_reach(thr)
        if ep is not None:
            results.append(("plateau", f"first_src_acc>={thr}", f"ep{ep}", tgt_at(ep)))
            for n in (5, 10):
                ep2 = min(ep + n, last_ep)
                results.append(("plateau", f"first>={thr}+{n}ep", f"ep{ep2}", tgt_at(ep2)))
    # middle of plateau & earliest stable plateau (m consecutive >= 0.99)
    plat = H[H["src_acc"] >= 0.99]["epoch"].to_numpy()
    if len(plat):
        mid = int(plat[len(plat) // 2])
        results.append(("plateau", "middle_of_plateau(0.99)", f"ep{mid}", tgt_at(mid)))
    for m_consec in (3, 5):
        accs = H["src_acc"].to_numpy(); ed = H["epoch"].to_numpy()
        found = None
        for i in range(len(accs) - m_consec + 1):
            if all(accs[i + j] >= 0.99 for j in range(m_consec)):
                found = int(ed[i]); break
        if found is not None:
            results.append(("plateau", f"earliest_stable_plateau(m={m_consec})", f"ep{found}", tgt_at(found)))

    # ---- robustness-val rules ----
    pick_by("aug_f1_mean", True, "best_augmented_val_f1", cat="robustness")
    pick_by("worst_aug_f1", True, "best_worst_aug_f1", cat="robustness")
    pick_by("aug_consistency", True, "best_aug_consistency", cat="robustness")
    pick_by("aug_entropy", False, "min_aug_entropy", cat="robustness")
    H["_clean_plus_aug"] = 0.5 * H["src_f1"] + 0.5 * H["aug_f1_mean"]
    pick_by("_clean_plus_aug", True, "clean+aug_combined(0.5/0.5)", cat="robustness")

    # ---- invariance rules ----
    pick_by("feat_cos_inv", True, "best_feature_stability(DAS)", cat="invariance")
    pick_by("pred_kl_inv", False, "min_prediction_KL(DAS)", cat="invariance")
    pick_by("carrier_fisher", False, "min_carrier_feature_separability", cat="invariance")

    # ---- hybrid scores ----
    sf1 = H["src_f1"].to_numpy()
    for lam in (0.5, 1.0):
        H[f"_h_kl_{lam}"] = sf1 - lam * H["pred_kl_inv"].to_numpy()
        pick_by(f"_h_kl_{lam}", True, f"src_f1 - {lam}*KL(aug)", cat="hybrid")
    H["_h_nll"] = sf1 - 0.1 * H["src_nll"].to_numpy()
    pick_by("_h_nll", True, "src_f1 - 0.1*NLL", cat="hybrid")
    H["_h_aug"] = sf1 + 0.5 * H["aug_f1_mean"].to_numpy()
    pick_by("_h_aug", True, "src_f1 + 0.5*aug_f1", cat="hybrid")
    # constrained: src_f1 >= 0.99 quantile region, then best aug-consistency
    cons_pool = H[H["src_f1"] >= H["src_f1"].quantile(0.90)]
    if len(cons_pool):
        i = int(cons_pool["aug_consistency"].to_numpy().argmax())
        ep = int(cons_pool["epoch"].iloc[i])
        results.append(("hybrid", "constrained: src_f1 top10% -> best aug_consistency", f"ep{ep}", tgt_at(ep)))

    # ---- oracle diagnostics ----
    results.append(("ORACLE", "target_best (upper bound)", f"ep{oracle_ep}", oracle_f1))

    # ===================================================================== #
    # Rank correlations: each source signal vs target F1 (over plateau region) #
    # ===================================================================== #
    from scipy.stats import spearmanr, kendalltau
    corr_region = H[H["src_acc"] >= 0.90].reset_index(drop=True)  # where selection operates
    if len(corr_region) < 5:   # partial/early data fallback -> use all EMA-valid epochs
        corr_region = H.reset_index(drop=True)
    sig_cols = ["src_acc", "src_f1", "src_bal_acc", "src_worst_class_f1",
                "src_worst_group_f1", "src_nll", "src_brier", "src_ece",
                "src_margin", "src_entropy", "src_hi_conf_err",
                "aug_f1_mean", "worst_aug_f1", "aug_consistency", "aug_entropy",
                "feat_cos_inv", "pred_kl_inv", "carrier_fisher"]
    corrs = []
    tgt_region = corr_region["tgt_f1"].to_numpy()
    for c in sig_cols:
        sp = float(spearmanr(corr_region[c], tgt_region).correlation)
        kt = float(kendalltau(corr_region[c], tgt_region).correlation)
        corrs.append({"signal": c, "spearman": sp, "kendall": kt, "abs_spearman": abs(sp)})
    corrs.sort(key=lambda r: -r["abs_spearman"])

    # ---- regret table ----
    table = []
    for cat, rule, pick, tf in results:
        table.append({"category": cat, "rule": rule, "picked": pick,
                      "tgt_f1": round(float(tf), 4),
                      "regret": round(float(oracle_f1 - tf), 4)})
    nonoracle = [t for t in table if t["category"] != "ORACLE"]
    nonoracle.sort(key=lambda t: t["regret"])

    summary = {
        "seed_run": str(run_dir), "basis": args.basis,
        "n_epochs": len(hist),
        "oracle_epoch": oracle_ep, "oracle_tgt_f1": round(oracle_f1, 4),
        "target_set_size": int(len(tgt_df)), "source_val_size": int(len(val_df)),
        "ranked_rules": nonoracle,
        "correlations": corrs,
        "corr_region_epochs": [int(corr_region["epoch"].min()), int(corr_region["epoch"].max())],
    }
    (out_dir / "diagnostic.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ---- markdown ----
    md = []
    md.append(f"# Checkpoint-selection diagnostic (seed run, basis={args.basis})\n")
    md.append(f"- Target set: full 77GHz = {len(tgt_df)} imgs (val merged into test). "
              f"Source-val = {len(val_df)} (10/24GHz).")
    md.append(f"- **ORACLE target-best = {oracle_f1:.4f} @ ep{oracle_ep}** "
              f"(max 77GHz macro-F1 over epochs; upper bound, not a usable rule).\n")
    md.append("## Selection rules ranked by regret (closest to oracle first)\n")
    md.append("| rank | category | rule | picked | 77GHz F1 | regret |")
    md.append("|---:|---|---|---|---:|---:|")
    for i, t in enumerate(nonoracle, 1):
        md.append(f"| {i} | {t['category']} | {t['rule']} | {t['picked']} | "
                  f"{t['tgt_f1']:.4f} | {t['regret']:+.4f} |")
    md.append("\n## Source-signal -> 77GHz F1 rank correlation (plateau region)\n")
    md.append(f"Region: epochs {corr_region['epoch'].min()}-{corr_region['epoch'].max()} "
              f"(source-val acc >= 0.90). Higher |Spearman| = more selection power.\n")
    md.append("| signal | Spearman | Kendall |")
    md.append("|---|---:|---:|")
    for c in corrs:
        md.append(f"| {c['signal']} | {c['spearman']:+.3f} | {c['kendall']:+.3f} |")
    (out_dir / "diagnostic.md").write_text("\n".join(md), encoding="utf-8")

    print("\n" + "\n".join(md[:40]))
    print(f"\n[diag] wrote {out_dir/'diagnostic.md'} and diagnostic.json")


if __name__ == "__main__":
    main()
