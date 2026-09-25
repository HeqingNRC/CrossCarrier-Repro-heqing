"""SPR-Morph decisive test (NO training): does the source-pair residual DIRECTION
that DAS misses (DAS(10->24) -> real-24) PREDICT the target gap direction
(DAS(src->77) -> real-77)? Measured in the carrier-NAIVE frozen DINOv3 oracle
(unconfounded by the LoRA training that was pushed toward 10<->24 invariance).

If per-class cosine(residual_dir, gap_dir) is high AND magnitudes are comparable,
the source residual extrapolates -> SPR-Morph has a path. If misaligned or the
source residual is tiny vs the target gap, the one source interval cannot
determine the law to 77 -> kill.
"""
from __future__ import annotations
import numpy as np, pandas as pd, torch
from pathlib import Path
from PIL import Image
import config, v9_2_1lib as lib

DEVICE = torch.device("cuda"); AMP = torch.bfloat16; NC = config.NUM_CLASSES
ROOT = config.DATASET_ROOT; C = config.CLASSES
_tf = lib.tensor_transform(train=False)


def oracle_feats(model, pil_list, bs=128):
    out = []
    for i in range(0, len(pil_list), bs):
        x = torch.stack([_tf(im) for im in pil_list[i:i + bs]]).to(DEVICE)
        with torch.no_grad(), torch.autocast("cuda", dtype=AMP):
            out.append(model.oracle_encoder(x).float().cpu())
    z = torch.cat(out).numpy()
    return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)


def centroid(z, y, c):
    m = z[y == c]
    if len(m) < 3:
        return None
    v = m.mean(0); return v / (np.linalg.norm(v) + 1e-8)


def main():
    tr = lib.load_manifest("train", keep_7c=True); va = lib.load_manifest("val", keep_7c=True)
    src = pd.concat([tr, va], ignore_index=True)
    pair_df, _ = lib.build_exact_pairs(src)
    yp = pair_df["class_idx_7c"].to_numpy()
    im10 = [Image.open(ROOT / p).convert("RGB") for p in pair_df["path_10"]]
    im24 = [Image.open(ROOT / p).convert("RGB") for p in pair_df["path_24"]]
    imdas24 = [lib.das_deterministic(im.copy(), 10.0, 24.0) for im in im10]

    tgt = lib.load_manifest("test", keep_7c=True); y77 = tgt["class_idx_7c"].to_numpy()
    im77 = [Image.open(ROOT / p).convert("RGB") for p in tgt["path"]]
    # DAS(source->77): push val images to 77 (same operator the realism gap uses)
    yv = va["class_idx_7c"].to_numpy()
    imval_das77 = [lib.das_deterministic(Image.open(ROOT / p).convert("RGB"),
                   lib.parse_freq_ghz(f), 77.0) for p, f in zip(va["path"], va["frequency"])]

    # frozen oracle (carrier-naive) is enough; build a model just to get oracle_encoder
    model = lib.TimmBackboneV921(config.DEFAULT_BACKBONE, NC, adapter_mode="lora").to(DEVICE).eval()
    z24 = oracle_feats(model, im24); zdas24 = oracle_feats(model, imdas24)
    z77 = oracle_feats(model, im77); zdas77 = oracle_feats(model, imval_das77)

    print("==== SPR-Morph alignment (carrier-naive oracle features) ====")
    print(f"{'class':9s} {'|src_resid|':>11} {'|tgt_gap|':>10} {'cos(resid,gap)':>15}")
    coss, ratios = [], []
    for ci, cc in enumerate(C):
        c24 = centroid(z24, yp, ci); cd24 = centroid(zdas24, yp, ci)       # real24, DAS(10->24)
        c77 = centroid(z77, y77, ci); cd77 = centroid(zdas77, yv, ci)      # real77, DAS(src->77)
        if any(v is None for v in (c24, cd24, c77, cd77)):
            print(f"{cc:9s}  (insufficient samples)"); continue
        resid = c24 - cd24                      # what DAS misses on the SEEN band
        gap = c77 - cd77                        # what DAS misses on the TARGET band
        nr, ng = np.linalg.norm(resid), np.linalg.norm(gap)
        cos = float(resid @ gap / (nr * ng + 1e-8))
        coss.append(cos); ratios.append(ng / (nr + 1e-8))
        print(f"{cc:9s} {nr:>11.3f} {ng:>10.3f} {cos:>15.3f}")
    print(f"\nmean cos(resid,gap) = {np.mean(coss):+.3f}   (≈0 -> source residual does NOT point at the 77 gap)")
    print(f"mean |gap|/|resid|  = {np.mean(ratios):.1f}x  (how far you must extrapolate one interval)")
    ok = np.mean(coss) > 0.5 and np.mean(ratios) < 3
    print("VERDICT SPR-Morph:", "POTENTIAL (residual aligns with & scales to the 77 gap)" if ok
          else "KILL/WEAK (one source interval cannot determine the 77 extrapolation: "
               "residual direction misaligned and/or magnitude far short of the gap)")


if __name__ == "__main__":
    main()
