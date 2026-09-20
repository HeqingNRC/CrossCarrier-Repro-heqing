"""Realism gate (verify_priorities V3) + head contribution (V2) for ONE checkpoint.

V3: real-77 vs DAS(source->77) linear discriminability in z_cls (0.5=identical,
1.0=fully separable = realism gap). Lower = synthetic falsify views look more like
real 77GHz. Also per-class centroid-distance excess (real vs das minus within-real
floor) for the weak classes. Reuses verify_priorities (same probe/cache as the
0.981 baseline it reports).

Usage: python probe_realism.py --ckpt <run>/checkpoints/pool_ep100_ema.pt [--tag NAME]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import config
import v9_2_1lib as lib
from verify_priorities import cache, extract, torch_cv_probe, accf1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
C = config.CLASSES
NC = config.NUM_CLASSES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    va = lib.load_manifest("val", keep_7c=True).reset_index(drop=True)
    tgt = lib.load_manifest("test", keep_7c=True).reset_index(drop=True)
    yv = va["class_idx_7c"].to_numpy()
    yt = tgt["class_idx_7c"].to_numpy()
    x_val = cache(va)
    x_tgt = cache(tgt)
    x_das77 = cache(va, das_to=77.0)            # source-val pushed to 77 via geometric DAS

    model = lib.TimmBackboneV921(config.DEFAULT_BACKBONE, NC, adapter_mode="lora").to(DEVICE).eval()
    ck = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    state = ck["ema"] if "ema" in ck else ck.get("model", ck)
    model.load_state_dict({k: v.to(DEVICE) for k, v in state.items()}, strict=False)

    np.random.seed(0)
    # ---- V3: realism gap ----
    zc_t, _, kin_t, sen_t = extract(model, x_tgt)
    zc_das, _, _, _ = extract(model, x_das77)
    zc_real = zc_t / (np.linalg.norm(zc_t, axis=1, keepdims=True) + 1e-8)
    zc_das = zc_das / (np.linalg.norm(zc_das, axis=1, keepdims=True) + 1e-8)
    Xd = np.concatenate([zc_real, zc_das])
    yd = np.r_[np.ones(len(zc_real)), np.zeros(len(zc_das))].astype(int)
    disc = torch_cv_probe(Xd, yd, 2)

    excess = {}
    for ci, cc in enumerate(C):
        r = zc_real[yt == ci]; d = zc_das[yv == ci]
        if len(r) >= 4 and len(d) >= 1:
            cr = r.mean(0); cr /= np.linalg.norm(cr) + 1e-8
            cd = d.mean(0); cd /= np.linalg.norm(cd) + 1e-8
            cent = float(1 - cr @ cd)
            h = np.random.permutation(len(r)); a = r[h[:len(r) // 2]]; b = r[h[len(r) // 2:]]
            ca = a.mean(0); ca /= np.linalg.norm(ca) + 1e-8
            cb = b.mean(0); cb /= np.linalg.norm(cb) + 1e-8
            floor = float(1 - ca @ cb)
            excess[cc] = round(cent - floor, 4)

    # ---- V2: head contribution on 77-418 ----
    kin418 = accf1(kin_t, yt)[1]
    tot418 = accf1(kin_t + sen_t, yt)[1]
    sen418 = accf1(sen_t, yt)[1]

    out = {
        "tag": args.tag, "ckpt": str(args.ckpt),
        "V3_real_vs_das77_discriminability": round(disc, 4),
        "V3_per_class_excess": excess,
        "V2_kin418_f1": round(kin418, 4), "V2_total418_f1": round(tot418, 4),
        "V2_sensor418_f1": round(sen418, 4),
    }
    print(json.dumps(out, indent=2))
    print(f"\n[REALISM {args.tag}] discriminability={disc:.3f} (0.5=ideal, 0.981=baseline gap) | "
          f"excess Sit={excess.get('Sit')} Bend={excess.get('Bend')} Towards={excess.get('Towards')}")


if __name__ == "__main__":
    main()
