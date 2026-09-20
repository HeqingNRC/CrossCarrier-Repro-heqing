"""5-row component ablation under the final-EMA honest selection rule.

Selection rule: take the EMA weights at the LAST epoch (ep100) -- no target
peeking, no source-val gate. Each run's pool_ep100_ema.pt holds exactly that.

Eval set: FULL 418 77GHz (val merged into test = the active protocol). We also
eval the 278-subset (the trainer's internal test77_final) purely to cross-check
our eval harness against each run's own history.json (must match ~exactly).

Rows (all configs already trained, 100ep x seeds 42/1234/31415, in REVISION_5090):
  Full (Proposed)            DAS + residual(V13) + falsify(V15)   = A_V20
  - falsification            DAS + residual                       = A_V13
  - residual split           DAS + falsify                        = A_V15
  - both modules (DAS only)  DAS                                  = A_REF
  - DAS (pure base)          (nothing)                            = E1_noDAS
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import config
import v9_2_1lib as lib

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = torch.bfloat16
NC = config.NUM_CLASSES
ROOT = config.DATASET_ROOT
REV = Path("EXPERIMENTSRESULT/REVISION_5090")

CONFIGS = [
    ("Full (Proposed)",          "A_V20"),
    ("- falsification",          "A_V13"),
    ("- residual split",         "A_V15"),
    ("- both modules (DAS only)", "A_REF"),
    ("- DAS (pure base)",        "E1_noDAS"),
    ("V13-as-GRL (shown w0.3)",  "A_V13_GRL"),
    ("GRL + wc-margin V15R",     "A_GRLWC"),
    ("DAS jitter (physics-free)", "E1_jitter"),
    ("DANN (data-driven)",        "E1_DANN"),
    ("full-method + jitter",      "E1B_Pjitter"),
    ("DAS fixed-full (no curric)", "E2_full"),
    ("DAS fixed-narrow",          "E2_narrow"),
]
SEEDS = [42, 1234, 31415]

# Supplementary-ablation runs (under REV/SUPP_ABLATION/<cfg>). Evaluated ONLY when
# AGG_INCLUDE_SUPP=1 so the default run stays byte-identical to the validated path.
# Final-EMA ckpt resolves to pool_ep100_ema.pt if present, else epoch_ckpts/ep100.pt
# (DUMP_EPOCH_CKPTS=1 guarantees the latter). Missing seeds are skipped with a note.
SUPP_CONFIGS = [
    ("ACR L_freq only",        "ACR_Lfreq_only"),
    ("ACR GRL only",           "ACR_GRL_only"),
    ("ACR discrete adversary", "ACR_discrete"),
    ("DAS ERM (no aug)",       "DAS_ERM"),
    ("DAS stretch only",       "DAS_stretch_only"),
    ("DAS fixed-full + ACR",   "SCHED_fixedfull_ACR"),
    ("DAS fixed-narrow + ACR", "SCHED_fixednarrow_ACR"),
]
TOTAL_EPOCHS = int(os.environ.get("AGG_TOTAL_EPOCHS", "100"))

_tf = lib.tensor_transform(train=False)


def resolve_ckpt(seed_dir: Path):
    """final-EMA ckpt: prefer pool_ep{N}_ema.pt, else epoch_ckpts/ep{N:03d}.pt."""
    pool = seed_dir / "checkpoints" / f"pool_ep{TOTAL_EPOCHS}_ema.pt"
    if pool.exists():
        return pool
    epc = seed_dir / "epoch_ckpts" / f"ep{TOTAL_EPOCHS:03d}.pt"
    if epc.exists():
        return epc
    return None


def cache(df):
    return torch.stack(
        [_tf(Image.open(ROOT / p).convert("RGB")) for p in df["path"]]
    ).to(DEVICE)


@torch.no_grad()
def eval_accf1(model, x, y, bs=256):
    preds = []
    for i in range(0, len(y), bs):
        with torch.autocast("cuda", dtype=AMP):
            z = model.encode_adapted(x[i:i + bs])
            lg = model.logits_from_neck(z, margin=False)
        preds.append(lg.float().argmax(1).cpu())
    preds = torch.cat(preds).numpy()
    y = np.asarray(y)
    acc = float((preds == y).mean())
    f1s = []
    for c in range(NC):
        tp = int(((preds == c) & (y == c)).sum())
        fp = int(((preds == c) & (y != c)).sum())
        fn = int(((preds != c) & (y == c)).sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * p * r / (p + r) if (p + r) else 0.0)
    return acc, float(np.mean(f1s))


def main():
    # ---- data: full 418 + 278-subset (for cross-check) ----
    full = lib.load_manifest("test", keep_7c=True).reset_index(drop=True)
    y418 = full["class_idx_7c"].to_numpy()
    x418 = cache(full)
    _dev, test278, _info = lib.split_test77_dev_final(full, dev_per_class=20, seed=42)
    test278 = test278.reset_index(drop=True)
    y278 = test278["class_idx_7c"].to_numpy()
    x278 = cache(test278)
    print(f"[data] full-418={len(full)}  278-subset={len(test278)}")

    model = lib.TimmBackboneV921(config.DEFAULT_BACKBONE, NC, adapter_mode="lora").to(DEVICE).eval()

    # base dir per config: anchors live at REV/<cfg>; SUPP runs at REV/SUPP_ABLATION/<cfg>
    eval_list = [(n, c, REV / c) for (n, c) in CONFIGS]
    if os.environ.get("AGG_INCLUDE_SUPP", "0") == "1":
        eval_list += [(n, c, REV / "SUPP_ABLATION" / c) for (n, c) in SUPP_CONFIGS]
        print(f"[supp] AGG_INCLUDE_SUPP=1 -> +{len(SUPP_CONFIGS)} supplementary configs")

    rows = []
    xcheck = []
    for name, cfg, base in eval_list:
        a418, f418, a278, f278 = [], [], [], []
        for s in SEEDS:
            seed_dir = base / f"seed{s}"
            ckpt = resolve_ckpt(seed_dir)
            if ckpt is None:
                print(f"[skip] {cfg} seed{s}: no final-EMA ckpt under {seed_dir}")
                continue
            ck = torch.load(ckpt, map_location=DEVICE, weights_only=False)
            model.load_state_dict({k: v.to(DEVICE) for k, v in ck["ema"].items()}, strict=False)
            acc418, mf418 = eval_accf1(model, x418, y418)
            acc278, mf278 = eval_accf1(model, x278, y278)
            a418.append(acc418); f418.append(mf418); a278.append(acc278); f278.append(mf278)
            # cross-check: our 278 final-EMA F1 vs run's history last-epoch test77_f1_ema
            hist_path = seed_dir / "reports" / "history.json"
            if hist_path.exists():
                hist = json.load(open(hist_path))
                hist_lastep = float(hist[-1]["test77_f1_ema"])
                xcheck.append((cfg, s, mf278, hist_lastep, mf278 - hist_lastep))
                print(f"{cfg:9s} seed{s:<5}: 418 acc/f1={acc418:.4f}/{mf418:.4f}  "
                      f"278 f1={mf278:.4f} (hist {hist_lastep:.4f}, d{mf278-hist_lastep:+.4f})")
            else:
                print(f"{cfg:9s} seed{s:<5}: 418 acc/f1={acc418:.4f}/{mf418:.4f}  "
                      f"278 f1={mf278:.4f} (no history.json)")
        if not f418:
            print(f"[skip] {cfg}: no seeds evaluated")
            continue
        rows.append({
            "variant": name, "config": cfg,
            "acc_mean": float(np.mean(a418)), "acc_std": float(np.std(a418)),
            "f1_mean": float(np.mean(f418)), "f1_std": float(np.std(f418)),
            "f1_seeds": [round(v, 4) for v in f418],
            "acc_seeds": [round(v, 4) for v in a418],
        })

    full_f1 = rows[0]["f1_mean"]
    for r in rows:
        r["delta_pp"] = round((r["f1_mean"] - full_f1) * 100, 1)

    # ---- output ----
    out = Path("EXPERIMENTSRESULT/CKPT_SELECTION_DIAG/ablation_finalEMA_3seed.json")
    out.write_text(json.dumps({"rows": rows, "xcheck": xcheck}, indent=2), encoding="utf-8")

    md = []
    md.append("# Component ablation -- final-EMA honest rule, full-418 77GHz, 3 seeds (42/1234/31415)\n")
    md.append("Selection = EMA weights at the last epoch (ep100); no target/source-val peeking.\n")
    md.append("| Variant | Accuracy | Macro-F1 | d macro-F1 (pp) |")
    md.append("|---|---|---|---:|")
    for r in rows:
        dpp = "--" if r["config"] == "A_V20" else f"{r['delta_pp']:+.1f}"
        md.append(f"| {r['variant']} | {r['acc_mean']:.3f} ± {r['acc_std']:.3f} | "
                  f"{r['f1_mean']:.3f} ± {r['f1_std']:.3f} | {dpp} |")
    md.append("\nPer-seed Macro-F1 (full-418):")
    md.append("\n| Variant | seed42 | seed1234 | seed31415 |")
    md.append("|---|---:|---:|---:|")
    for r in rows:
        md.append(f"| {r['variant']} | " + " | ".join(f"{v:.4f}" for v in r["f1_seeds"]) + " |")
    md.append("\n## Component contributions (full-418 macro-F1, mean)")
    g = {r["config"]: r["f1_mean"] for r in rows}

    def _have(*keys):
        return all(k in g for k in keys)

    # Each line emits only when its required configs were actually evaluated, so the
    # summary degrades gracefully under partial / SUPP-only runs.
    if _have("A_REF", "E1_noDAS"):
        md.append(f"- DAS (foundation): A_REF {g['A_REF']:.3f} - E1_noDAS {g['E1_noDAS']:.3f} "
                  f"= **{(g['A_REF']-g['E1_noDAS'])*100:+.1f} pp**")
    if _have("A_V20", "A_V13"):
        md.append(f"- falsification (V15) on top of full: Full - (-falsif) "
                  f"= **{(g['A_V20']-g['A_V13'])*100:+.1f} pp**")
    if _have("A_V20", "A_V15"):
        md.append(f"- residual split (V13) on top of full: Full - (-residual) "
                  f"= **{(g['A_V20']-g['A_V15'])*100:+.1f} pp**")
    if _have("A_V15", "A_REF"):
        md.append(f"- V15 alone over DAS-base: A_V15 - A_REF = **{(g['A_V15']-g['A_REF'])*100:+.1f} pp**")
    if _have("A_V13", "A_REF"):
        md.append(f"- V13 alone over DAS-base: A_V13 - A_REF = **{(g['A_V13']-g['A_REF'])*100:+.1f} pp**")
    if _have("A_V20", "A_V13", "A_V15", "A_REF"):
        md.append(f"- both modules over DAS-base: A_V20 - A_REF = **{(g['A_V20']-g['A_REF'])*100:+.1f} pp** "
                  f"(vs sum of singles {((g['A_V13']-g['A_REF'])+(g['A_V15']-g['A_REF']))*100:+.1f} pp "
                  f"-> additivity/complementarity check)")
    # SUPP component deltas (only when those configs were evaluated)
    if _have("A_V13_GRL", "ACR_Lfreq_only"):
        md.append(f"- ACR GRL half: A_V13_GRL - ACR_Lfreq_only "
                  f"= **{(g['A_V13_GRL']-g['ACR_Lfreq_only'])*100:+.1f} pp**")
    if _have("A_V13_GRL", "ACR_GRL_only"):
        md.append(f"- ACR residual half: A_V13_GRL - ACR_GRL_only "
                  f"= **{(g['A_V13_GRL']-g['ACR_GRL_only'])*100:+.1f} pp**")
    if _have("A_V13_GRL", "ACR_discrete"):
        md.append(f"- continuous vs discrete adversary: A_V13_GRL - ACR_discrete "
                  f"= **{(g['A_V13_GRL']-g['ACR_discrete'])*100:+.1f} pp**")
    if _have("DAS_stretch_only", "DAS_ERM"):
        md.append(f"- Doppler-stretch alone over ERM: DAS_stretch_only - DAS_ERM "
                  f"= **{(g['DAS_stretch_only']-g['DAS_ERM'])*100:+.1f} pp**")
    md.append("\n## Eval-harness cross-check (278-subset final-EMA vs run history last-epoch)")
    maxd = max((abs(d) for *_, d in xcheck), default=0.0)
    md.append(f"max |diff| vs history.json = {maxd:.4f} (should be ~0 -> harness validated)")
    text = "\n".join(md)
    Path("EXPERIMENTSRESULT/CKPT_SELECTION_DIAG/ablation_finalEMA_3seed.md").write_text(text, encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()
