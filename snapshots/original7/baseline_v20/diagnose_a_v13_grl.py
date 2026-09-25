"""Diagnose A_V13_GRL (SOTA 0.832): where do the errors / variance come from?

final-EMA (pool_ep100_ema.pt), full-418 77GHz, seeds 42/1234/31415, kin-only head
(the validated inference path). Reports:
  1. per-class precision/recall/F1 (3-seed mean) -> FP- vs FN-bound classes
  2. aggregate confusion matrix (summed over seeds) -> top off-diagonal confusions
  3. per-seed per-class F1 -> which classes drive the 0.034 cross-seed variance
  4. 3-seed logit-ensemble F1 -> is a free ensemble above the 0.832 mean?
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
_tf = lib.tensor_transform(train=False)


def cache(df):
    return torch.stack([_tf(Image.open(ROOT / p).convert("RGB")) for p in df["path"]]).to(DEVICE)


@torch.no_grad()
def logits(model, x, bs=256):
    out = []
    for i in range(0, x.shape[0], bs):
        with torch.autocast("cuda", dtype=AMP):
            z = model.encode_adapted(x[i:i + bs])
            lg = model.logits_from_neck(z, margin=False)
        out.append(lg.float().cpu())
    return torch.cat(out).numpy()


def prf(preds, y):
    P, R, F = {}, {}, {}
    for c in range(NC):
        tp = int(((preds == c) & (y == c)).sum()); fp = int(((preds == c) & (y != c)).sum())
        fn = int(((preds != c) & (y == c)).sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0; r = tp / (tp + fn) if (tp + fn) else 0.0
        P[C[c]] = p; R[C[c]] = r; F[C[c]] = 2 * p * r / (p + r) if (p + r) else 0.0
    return P, R, F


def main():
    full = lib.load_manifest("test", keep_7c=True).reset_index(drop=True)
    y = full["class_idx_7c"].to_numpy(); x = cache(full)
    counts = {C[c]: int((y == c).sum()) for c in range(NC)}
    print(f"[data] full-418={len(full)} class counts={counts}")
    model = lib.TimmBackboneV921(config.DEFAULT_BACKBONE, NC, adapter_mode="lora").to(DEVICE).eval()

    all_logits, per_seed_f1 = [], {}
    conf = np.zeros((NC, NC), dtype=int)        # summed over seeds: conf[true, pred]
    pcls = {cc: [] for cc in C}
    for s in SEEDS:
        ck = torch.load(REV / "A_V13_GRL" / f"seed{s}" / "checkpoints" / "pool_ep100_ema.pt",
                        map_location=DEVICE, weights_only=False)
        model.load_state_dict({k: v.to(DEVICE) for k, v in ck["ema"].items()}, strict=False)
        lg = logits(model, x); all_logits.append(lg)
        preds = lg.argmax(1)
        _, _, F = prf(preds, y)
        per_seed_f1[s] = float(np.mean(list(F.values())))
        for cc in C: pcls[cc].append(F[cc])
        for t, p in zip(y, preds): conf[t, p] += 1

    # ---- 1. per-class P/R/F1 (3-seed mean) ----
    Ps, Rs, Fs = [], [], []
    for s, lg in zip(SEEDS, all_logits):
        P, R, F = prf(lg.argmax(1), y); Ps.append(P); Rs.append(R); Fs.append(F)
    print("\n==== per-class precision / recall / F1 (3-seed mean) ====")
    print(f"{'class':9s} {'prec':>7} {'recall':>7} {'F1':>7}  (n)")
    for cc in C:
        p = np.mean([d[cc] for d in Ps]); r = np.mean([d[cc] for d in Rs]); f = np.mean([d[cc] for d in Fs])
        print(f"{cc:9s} {p:>7.3f} {r:>7.3f} {f:>7.3f}  ({counts[cc]})")

    # ---- 2. top confusions (off-diagonal, normalized by true-class support over 3 seeds) ----
    print("\n==== top confusions (summed over 3 seeds; frac of that true class) ====")
    pairs = []
    for t in range(NC):
        tot = conf[t].sum()
        for p in range(NC):
            if t != p and conf[t, p] > 0:
                pairs.append((conf[t, p] / tot, C[t], C[p], conf[t, p]))
    for frac, tc, pc, n in sorted(pairs, reverse=True)[:10]:
        print(f"  {tc:8s} -> {pc:8s}  {frac:5.1%}  (n={n})")

    # ---- 3. per-seed per-class F1 -> variance source ----
    print("\n==== per-seed per-class F1 (variance source; std across seeds) ====")
    print(f"{'class':9s} " + " ".join(f"s{s:<6}" for s in SEEDS) + "  std")
    for cc in C:
        vals = pcls[cc]
        print(f"{cc:9s} " + " ".join(f"{v:>7.3f}" for v in vals) + f"  {np.std(vals):.3f}")
    print(f"{'MACRO':9s} " + " ".join(f"{per_seed_f1[s]:>7.3f}" for s in SEEDS) +
          f"  {np.std(list(per_seed_f1.values())):.3f}")

    # ---- 4. 3-seed logit ensemble ----
    ens = np.mean(all_logits, axis=0).argmax(1)
    _, _, Fe = prf(ens, y)
    ens_f1 = float(np.mean(list(Fe.values())))
    ens_acc = float((ens == y).mean())
    print(f"\n==== 3-seed logit ENSEMBLE: acc={ens_acc:.4f} macro-F1={ens_f1:.4f} "
          f"(vs single-seed mean {np.mean(list(per_seed_f1.values())):.4f}) ====")

    Path("EXPERIMENTSRESULT/CKPT_SELECTION_DIAG").mkdir(parents=True, exist_ok=True)
    Path("EXPERIMENTSRESULT/CKPT_SELECTION_DIAG/diagnose_a_v13_grl.json").write_text(json.dumps({
        "per_seed_macro_f1": per_seed_f1,
        "per_class_f1_mean": {cc: float(np.mean(pcls[cc])) for cc in C},
        "per_class_f1_std": {cc: float(np.std(pcls[cc])) for cc in C},
        "confusion": conf.tolist(),
        "ensemble_f1": ens_f1, "ensemble_acc": ens_acc,
    }, indent=2), encoding="utf-8")
    print("\n[done] wrote diagnose_a_v13_grl.json")


if __name__ == "__main__":
    main()
