"""Carrier-leakage probe for a SINGLE checkpoint (GRL tuning / verification).

Reuses verify_priorities' linear-probe machinery (torch_cv_probe / extract /
cache). For a given pool_ep*_ema.pt it reports the primary success gate:

  z_cls  -> carrier probe acc  (the thing GRL must drive 0.999 -> ~chance)
  z_freq -> carrier probe acc  (should stay HIGH; carrier neck still absorbs it)
  source-val macro-F1          (read from the ckpt payload; must not collapse)
  77GHz full-418 acc / macro-F1 (final-EMA eval, same harness as aggregate)

Usage:
  python probe_carrier_grl.py --ckpt <run>/checkpoints/pool_ep100_ema.pt [--tag NAME]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

import config
import v9_2_1lib as lib
from verify_priorities import torch_cv_probe, extract, cache

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = torch.bfloat16
NC = config.NUM_CLASSES
ROOT = config.DATASET_ROOT
_tf = lib.tensor_transform(train=False)


@torch.no_grad()
def eval_full418(model, x, y, bs=256):
    preds = []
    for i in range(0, len(y), bs):
        with torch.autocast("cuda", dtype=AMP):
            z = model.encode_adapted(x[i:i + bs])
            lg = model.logits_from_neck(z, margin=False)
        preds.append(lg.float().argmax(1).cpu())
    preds = torch.cat(preds).numpy()
    y = np.asarray(y)
    acc = float((preds == y).mean())
    f1s, per_class = [], {}
    for c in range(NC):
        tp = int(((preds == c) & (y == c)).sum())
        fp = int(((preds == c) & (y != c)).sum())
        fn = int(((preds != c) & (y == c)).sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        f1s.append(f1)
        per_class[config.CLASSES[c]] = round(f1, 4)
    return acc, float(np.mean(f1s)), per_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    ckpt_path = Path(args.ckpt)

    # --- source data + carrier labels (10 vs 24) ---
    tr = lib.load_manifest("train", keep_7c=True).reset_index(drop=True)
    va = lib.load_manifest("val", keep_7c=True).reset_index(drop=True)
    src = pd.concat([tr, va], ignore_index=True)
    carrier = (src["frequency"] == "24GHz").astype(int).to_numpy()
    x_src = cache(src)

    # --- target full-418 ---
    tgt = lib.load_manifest("test", keep_7c=True).reset_index(drop=True)
    yt = tgt["class_idx_7c"].to_numpy()
    x_tgt = cache(tgt)

    model = lib.TimmBackboneV921(config.DEFAULT_BACKBONE, NC, adapter_mode="lora").to(DEVICE).eval()
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state = ck["ema"] if "ema" in ck else ck.get("model", ck)
    model.load_state_dict({k: v.to(DEVICE) for k, v in state.items()}, strict=False)

    np.random.seed(0)  # deterministic CV folds
    zc, zf, _, _ = extract(model, x_src)
    probe_zcls = torch_cv_probe(zc, carrier, 2)
    probe_zfreq = torch_cv_probe(zf, carrier, 2)

    acc418, f1_418, per_class = eval_full418(model, x_tgt, yt)

    src_val_f1 = float(ck.get("source_val_f1_ema", -1.0))
    src_val_acc = float(ck.get("source_val_acc_ema", -1.0))
    chance = float(max(carrier.mean(), 1 - carrier.mean()))

    out = {
        "tag": args.tag,
        "ckpt": str(ckpt_path),
        "probe_zcls_carrier": round(probe_zcls, 4),
        "probe_zfreq_carrier": round(probe_zfreq, 4),
        "carrier_chance": round(chance, 4),
        "source_val_acc_ema_fromckpt": round(src_val_acc, 4),
        "source_val_f1_ema_fromckpt": round(src_val_f1, 4),
        "full418_acc": round(acc418, 4),
        "full418_macro_f1": round(f1_418, 4),
        "full418_per_class_f1": per_class,
    }
    print(json.dumps(out, indent=2))
    print(
        f"\n[PROBE {args.tag}] z_cls->carrier={probe_zcls:.3f} "
        f"z_freq->carrier={probe_zfreq:.3f} (chance~{chance:.3f}) | "
        f"src_val_f1(ckpt)={src_val_f1:.3f} | 77GHz-418 acc/f1={acc418:.3f}/{f1_418:.3f}"
    )


if __name__ == "__main__":
    main()
