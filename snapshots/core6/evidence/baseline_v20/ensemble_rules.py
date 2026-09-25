"""P0-C : ensemble-fairness eval (NO training; reuses existing checkpoints).

For each config, compute on the FULL-418 77GHz target set, using the validated
final-EMA inference path (kin-only head, pool_ep100_ema.pt or epoch_ckpts/ep100.pt):
  - single-seed mean +/- std macro-F1 / accuracy (3 seeds 42/1234/31415)
  - 3-seed ensemble under THREE rules:
        majority   : per-clip majority vote of the 3 per-seed argmaxes
                     (3-way tie -> broken by logit-avg argmax)
        logit-avg  : argmax of the mean RAW logits   (what diagnose_a_v13_grl used)
        posterior  : argmax of the mean SOFTMAX posteriors

Shows the +2.5 pp ensemble gain is not ACR-specific magic and posterior-averaging
is not a target-tuned trick (all three rules land in the same place).

Also DUMPS per-seed logits to eval_out/logits_cache.npz so paired_bootstrap_ci.py
can reuse them without re-running the backbone.

Selection rule = EMA weights at the LAST epoch (ep100); no target/source-val peek.
Outputs are written under EXPERIMENTSRESULT/REVISION_5090/SUPP_ABLATION/eval_out/.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import config
import v9_2_1lib as lib

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = torch.bfloat16
C = config.CLASSES
NC = config.NUM_CLASSES
ROOT = config.DATASET_ROOT
REV = Path("EXPERIMENTSRESULT/REVISION_5090")
SEEDS = [42, 1234, 31415]

# configs to evaluate (all are existing [HAVE] anchors -> no training).
#   (display name, dir under REV, subdir "")  -- subdir "SUPP_ABLATION" for new runs.
CONFIGS = [
    ("DAS only (A_REF)",       "A_REF",     ""),
    ("DAS+ACR (A_V13_GRL)",    "A_V13_GRL", ""),
    ("DAS jitter (E1_jitter)", "E1_jitter", ""),
]

OUT = REV / "SUPP_ABLATION" / "eval_out"
_tf = lib.tensor_transform(train=False)


def cache(df):
    return torch.stack(
        [_tf(Image.open(ROOT / p).convert("RGB")) for p in df["path"]]
    ).to(DEVICE)


def resolve_ckpt(seed_dir: Path, total_epochs: int = 100) -> Path | None:
    """final-EMA checkpoint: prefer pool_ep{N}_ema.pt, else epoch_ckpts/ep{N:03d}.pt."""
    pool = seed_dir / "checkpoints" / f"pool_ep{total_epochs}_ema.pt"
    if pool.exists():
        return pool
    epc = seed_dir / "epoch_ckpts" / f"ep{total_epochs:03d}.pt"
    if epc.exists():
        return epc
    return None


@torch.no_grad()
def get_logits(model, x, bs=256):
    out = []
    for i in range(0, x.shape[0], bs):
        with torch.autocast("cuda", dtype=AMP):
            z = model.encode_adapted(x[i:i + bs])
            lg = model.logits_from_neck(z, margin=False)
        out.append(lg.float().cpu())
    return torch.cat(out).numpy()


def macro_f1_acc(preds, y):
    preds = np.asarray(preds); y = np.asarray(y)
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


def majority_vote(per_seed_preds, logit_avg):
    """per_seed_preds: [S, N]; tie (all S differ) -> logit_avg argmax."""
    S, N = per_seed_preds.shape
    out = np.empty(N, dtype=int)
    la = logit_avg.argmax(1)
    for i in range(N):
        vals, cnts = np.unique(per_seed_preds[:, i], return_counts=True)
        top = cnts.max()
        winners = vals[cnts == top]
        out[i] = int(winners[0]) if len(winners) == 1 else int(la[i])
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    full = lib.load_manifest("test", keep_7c=True).reset_index(drop=True)
    y = full["class_idx_7c"].to_numpy()
    x = cache(full)
    print(f"[data] full-418={len(full)}")
    model = lib.TimmBackboneV921(
        config.DEFAULT_BACKBONE, NC, adapter_mode="lora"
    ).to(DEVICE).eval()

    rows = []
    logits_cache = {"y": y, "classes": np.array(C)}
    for name, cfg, subdir in CONFIGS:
        base = (REV / subdir / cfg) if subdir else (REV / cfg)
        seed_logits = []
        for s in SEEDS:
            ck_path = resolve_ckpt(base / f"seed{s}")
            if ck_path is None:
                print(f"[skip] {cfg} seed{s}: no final-EMA checkpoint found")
                continue
            ck = torch.load(ck_path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(
                {k: v.to(DEVICE) for k, v in ck["ema"].items()}, strict=False
            )
            seed_logits.append(get_logits(model, x))
        if len(seed_logits) < 2:
            print(f"[skip] {cfg}: <2 seeds available, cannot ensemble")
            continue
        L = np.stack(seed_logits, axis=0)            # [S, N, NC]
        logits_cache[cfg] = L
        per_seed_preds = L.argmax(2)                 # [S, N]

        single = [macro_f1_acc(per_seed_preds[i], y) for i in range(L.shape[0])]
        s_acc = np.array([a for a, _ in single]); s_f1 = np.array([f for _, f in single])

        logit_avg = L.mean(0)                        # [N, NC]
        post_avg = torch.softmax(torch.from_numpy(L), dim=2).mean(0).numpy()
        maj = majority_vote(per_seed_preds, logit_avg)
        e_maj = macro_f1_acc(maj, y)
        e_lg = macro_f1_acc(logit_avg.argmax(1), y)
        e_po = macro_f1_acc(post_avg.argmax(1), y)

        rows.append({
            "name": name, "config": cfg, "n_seeds": int(L.shape[0]),
            "single_f1_mean": float(s_f1.mean()), "single_f1_std": float(s_f1.std()),
            "single_acc_mean": float(s_acc.mean()), "single_acc_std": float(s_acc.std()),
            "single_f1_seeds": [round(v, 4) for v in s_f1.tolist()],
            "ens_majority_f1": e_maj[1], "ens_majority_acc": e_maj[0],
            "ens_logitavg_f1": e_lg[1], "ens_logitavg_acc": e_lg[0],
            "ens_posterior_f1": e_po[1], "ens_posterior_acc": e_po[0],
        })
        print(f"{name:24s} single={s_f1.mean():.4f}+/-{s_f1.std():.4f}  "
              f"maj={e_maj[1]:.4f}  logit-avg={e_lg[1]:.4f}  posterior={e_po[1]:.4f}")

    np.savez(OUT / "logits_cache.npz", **logits_cache)
    (OUT / "ensemble_rules.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    md = ["# P0-C ensemble fairness (PRELIMINARY -- eval-only, existing checkpoints)\n",
          "Full-418 77GHz, final-EMA, seeds 42/1234/31415. Single = per-seed mean+/-std.\n",
          "| Config | Single macro-F1 | Ens majority | Ens logit-avg | Ens posterior |",
          "|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['name']} | {r['single_f1_mean']:.4f} ± {r['single_f1_std']:.4f} "
                  f"| {r['ens_majority_f1']:.4f} | {r['ens_logitavg_f1']:.4f} "
                  f"| {r['ens_posterior_f1']:.4f} |")
    md.append("\n_Accuracy and per-seed values in ensemble_rules.json. Logits cached in "
              "logits_cache.npz for paired_bootstrap_ci.py._")
    (OUT / "ensemble_rules.md").write_text("\n".join(md), encoding="utf-8")
    print("\n[done] wrote", OUT / "ensemble_rules.md")


if __name__ == "__main__":
    main()
