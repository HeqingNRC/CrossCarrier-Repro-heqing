"""Phase 3 paired analysis: does the redesigned falsification (worst-case carrier
+ hardest-negative margin, realism OFF) stack on the GRL residual split?

Per seed, final-EMA (pool_ep100_ema.pt), full-418 77GHz:
  Delta = A_GRLWC - A_V13_GRL      (V15R worst-case+margin over the GRL-V13 base)
Goal: Delta 3/3 seeds positive AND mean macro-F1 above the A_V13_GRL base (0.832).
Also per-class F1 vs A_V13_GRL (esp. Sit/Bend). Honest note: realism is dropped
(falsified separately); this measures only the margin-based sub-mechanism.
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
CFGS = ["A_REF", "A_V13_GRL", "A_GRLWC"]
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
    for cfg in CFGS:
        fseeds = " ".join(f"{res[cfg]['f1'][s]:.4f}" for s in SEEDS)
        print(f"{cfg:12s} acc {mean(cfg,'acc'):.4f}±{std(cfg,'acc'):.4f}   "
              f"F1 {mean(cfg,'f1'):.4f}±{std(cfg,'f1'):.4f}   [{fseeds}]")

    print("\n================ Paired: A_GRLWC - A_V13_GRL (per seed) ================")
    d = []
    print(f"{'seed':>8} {'A_V13_GRL':>10} {'A_GRLWC':>9} {'Delta':>9}")
    for s in SEEDS:
        base = res["A_V13_GRL"]["f1"][s]; new = res["A_GRLWC"]["f1"][s]
        d.append(new - base)
        print(f"{s:>8} {base:>10.4f} {new:>9.4f} {new-base:>+9.4f}")
    npos = sum(1 for v in d if v > 0)
    base_mean = mean("A_V13_GRL", "f1"); new_mean = mean("A_GRLWC", "f1")
    print(f"{'mean':>8} {base_mean:>10.4f} {new_mean:>9.4f} {np.mean(d):>+9.4f}")
    print(f"\nDelta mean {np.mean(d):+.4f}; {npos}/3 seeds positive; "
          f"A_GRLWC mean {new_mean:.4f} vs base {base_mean:.4f}")
    verdict = ("REAL contributor (3/3 positive AND mean above base)"
               if (npos == 3 and new_mean > base_mean)
               else f"NOT a robust contributor ({npos}/3 positive, mean {'>' if new_mean>base_mean else '<='} base)")
    print(f"VERDICT (margin sub-mechanism): {verdict}")

    print("\n================ Per-class F1 (mean over seeds): A_GRLWC vs A_V13_GRL ================")
    print(f"{'class':10s} {'A_V13_GRL':>10} {'A_GRLWC':>9} {'delta':>8}")
    for c in config.CLASSES:
        b = float(np.mean([res["A_V13_GRL"]["per"][s][c] for s in SEEDS]))
        n = float(np.mean([res["A_GRLWC"]["per"][s][c] for s in SEEDS]))
        print(f"{c:10s} {b:>10.4f} {n:>9.4f} {n-b:>+8.4f}")

    out = {"f1": {c: {str(s): res[c]["f1"][s] for s in SEEDS} for c in CFGS},
           "delta_grlwc_vs_grl": {str(s): d[i] for i, s in enumerate(SEEDS)},
           "delta_mean": float(np.mean(d)), "npos": npos,
           "grlwc_mean": new_mean, "grl_base_mean": base_mean}
    Path("EXPERIMENTSRESULT/CKPT_SELECTION_DIAG").mkdir(parents=True, exist_ok=True)
    Path("EXPERIMENTSRESULT/CKPT_SELECTION_DIAG/v15r_paired_analysis.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print("\n[done] wrote v15r_paired_analysis.json")


if __name__ == "__main__":
    main()
