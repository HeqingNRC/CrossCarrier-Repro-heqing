"""Phase 3 paired analysis: is the V13-as-GRL (shown-target) contribution real
(non-noise) and larger than the old decorrelation V13?

For each seed, final-EMA (pool_ep100_ema.pt), full-418 77GHz:
  Delta_new = A_V13_GRL - A_REF      (new mechanism over DAS base)
  Delta_old = A_V13     - A_REF      (old decorr V13 over DAS base)
Report per-seed + mean. Goal: Delta_new consistently positive across all 3 seeds
(no sign flips) and larger than Delta_old. Also per-class A_V13_GRL vs A_V13.
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
NC = config.NUM_CLASSES
ROOT = config.DATASET_ROOT
REV = Path("EXPERIMENTSRESULT/REVISION_5090")
SEEDS = [42, 1234, 31415]
CFGS = ["A_REF", "A_V13", "A_V13_GRL"]
_tf = lib.tensor_transform(train=False)


def cache(df):
    return torch.stack([_tf(Image.open(ROOT / p).convert("RGB")) for p in df["path"]]).to(DEVICE)


@torch.no_grad()
def eval_accf1(model, x, y, bs=256):
    preds = []
    for i in range(0, len(y), bs):
        with torch.autocast("cuda", dtype=AMP):
            z = model.encode_adapted(x[i:i + bs])
            lg = model.logits_from_neck(z, margin=False)
        preds.append(lg.float().argmax(1).cpu())
    preds = torch.cat(preds).numpy(); y = np.asarray(y)
    acc = float((preds == y).mean())
    f1s, per = [], {}
    for c in range(NC):
        tp = int(((preds == c) & (y == c)).sum()); fp = int(((preds == c) & (y != c)).sum())
        fn = int(((preds != c) & (y == c)).sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0; r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        f1s.append(f1); per[config.CLASSES[c]] = f1
    return acc, float(np.mean(f1s)), per


def main():
    full = lib.load_manifest("test", keep_7c=True).reset_index(drop=True)
    y = full["class_idx_7c"].to_numpy(); x = cache(full)
    print(f"[data] full-418={len(full)}")
    model = lib.TimmBackboneV921(config.DEFAULT_BACKBONE, NC, adapter_mode="lora").to(DEVICE).eval()

    res = {c: {"acc": {}, "f1": {}, "per": {}} for c in CFGS}
    for cfg in CFGS:
        for s in SEEDS:
            ck = torch.load(REV / cfg / f"seed{s}" / "checkpoints" / "pool_ep100_ema.pt",
                            map_location=DEVICE, weights_only=False)
            model.load_state_dict({k: v.to(DEVICE) for k, v in ck["ema"].items()}, strict=False)
            a, f, per = eval_accf1(model, x, y)
            res[cfg]["acc"][s] = a; res[cfg]["f1"][s] = f; res[cfg]["per"][s] = per
            print(f"  {cfg:10s} seed{s:<6} acc/f1 = {a:.4f}/{f:.4f}")

    def mean(cfg, k): return float(np.mean([res[cfg][k][s] for s in SEEDS]))
    def std(cfg, k):  return float(np.std([res[cfg][k][s] for s in SEEDS]))

    print("\n================ Performance (final-EMA, full-418, 3 seeds) ================")
    print(f"{'config':12s} {'acc mean±std':>16s} {'F1 mean±std':>16s}  per-seed F1")
    for cfg in CFGS:
        fseeds = " ".join(f"{res[cfg]['f1'][s]:.4f}" for s in SEEDS)
        print(f"{cfg:12s} {mean(cfg,'acc'):.4f}±{std(cfg,'acc'):.4f}   "
              f"{mean(cfg,'f1'):.4f}±{std(cfg,'f1'):.4f}   [{fseeds}]")

    print("\n================ Paired analysis vs A_REF (per seed) ================")
    dnew, dold = [], []
    print(f"{'seed':>8} {'A_REF':>8} {'A_V13':>8} {'A_V13_GRL':>10} {'D_old':>8} {'D_new':>8}")
    for s in SEEDS:
        ref = res["A_REF"]["f1"][s]; old = res["A_V13"]["f1"][s]; new = res["A_V13_GRL"]["f1"][s]
        do = old - ref; dn = new - ref; dold.append(do); dnew.append(dn)
        print(f"{s:>8} {ref:>8.4f} {old:>8.4f} {new:>10.4f} {do:>+8.4f} {dn:>+8.4f}")
    print(f"{'mean':>8} {mean('A_REF','f1'):>8.4f} {mean('A_V13','f1'):>8.4f} "
          f"{mean('A_V13_GRL','f1'):>10.4f} {np.mean(dold):>+8.4f} {np.mean(dnew):>+8.4f}")
    npos_new = sum(1 for d in dnew if d > 0); npos_old = sum(1 for d in dold if d > 0)
    print(f"\nD_new (GRL-REF): mean {np.mean(dnew):+.4f}, {npos_new}/3 seeds positive")
    print(f"D_old (V13-REF): mean {np.mean(dold):+.4f}, {npos_old}/3 seeds positive")
    verdict = ("REAL & LARGER than old V13" if (npos_new == 3 and np.mean(dnew) > np.mean(dold))
               else "NOT robustly better than old V13")
    print(f"VERDICT: V13-as-GRL contribution is {verdict}")

    print("\n================ Per-class F1 (mean over seeds): A_V13_GRL vs A_V13 ================")
    print(f"{'class':10s} {'A_V13':>8} {'A_V13_GRL':>10} {'delta':>8}")
    for c in config.CLASSES:
        v13 = float(np.mean([res["A_V13"]["per"][s][c] for s in SEEDS]))
        grl = float(np.mean([res["A_V13_GRL"]["per"][s][c] for s in SEEDS]))
        print(f"{c:10s} {v13:>8.4f} {grl:>10.4f} {grl-v13:>+8.4f}")

    out = {"f1": {c: {str(s): res[c]["f1"][s] for s in SEEDS} for c in CFGS},
           "acc": {c: {str(s): res[c]["acc"][s] for s in SEEDS} for c in CFGS},
           "delta_new_vs_ref": {str(s): dnew[i] for i, s in enumerate(SEEDS)},
           "delta_old_vs_ref": {str(s): dold[i] for i, s in enumerate(SEEDS)},
           "delta_new_mean": float(np.mean(dnew)), "delta_old_mean": float(np.mean(dold)),
           "npos_new": npos_new, "npos_old": npos_old}
    Path("EXPERIMENTSRESULT/CKPT_SELECTION_DIAG").mkdir(parents=True, exist_ok=True)
    Path("EXPERIMENTSRESULT/CKPT_SELECTION_DIAG/grl_paired_analysis.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print("\n[done] wrote grl_paired_analysis.json")


if __name__ == "__main__":
    main()
