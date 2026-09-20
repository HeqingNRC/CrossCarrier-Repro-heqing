"""Where can the V13/V15 line still gain? Evidence-based per-class diagnosis.

For the Full model (A_V20), 3 seeds, final-EMA:
  1. per-class precision/recall/F1 on 77GHz-418 (target)  -> who is weak
  2. confusion matrix on 77GHz-418 (summed over seeds)    -> what they confuse with
  3. per-class F1 on SOURCE-val 10/24GHz (in-distribution) -> intrinsic vs cross-carrier
  4. cross-carrier gap per class = source_F1 - target_F1   -> who fails ONLY at 77GHz
  5. per-class carrier-fragility: agreement & feature-cosine between a source clip
     and its DAS->77GHz view -> which classes' decision flips under the carrier shift
     (this is exactly the feat_cos_inv / pred-KL signal that tracks 77GHz F1).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import config
import v9_2_1lib as lib

DEVICE = torch.device("cuda")
AMP = torch.bfloat16
C = config.CLASSES
NC = config.NUM_CLASSES
ROOT = config.DATASET_ROOT
REV = Path("EXPERIMENTSRESULT/REVISION_5090")
SEEDS = [42, 1234, 31415]
_tf = lib.tensor_transform(train=False)


def cache(df, das_to=None):
    imgs = []
    fsrc = [lib.parse_freq_ghz(f) for f in df["frequency"]]
    for p, fs in zip(df["path"], fsrc):
        im = Image.open(ROOT / p).convert("RGB")
        if das_to is not None and abs(das_to - fs) > 1e-9:
            im = lib.das_deterministic(im, fs, das_to)
        imgs.append(_tf(im))
    return torch.stack(imgs).to(DEVICE)


@torch.no_grad()
def fwd(model, x, bs=256):
    lg, zz = [], []
    for i in range(0, x.shape[0], bs):
        with torch.autocast("cuda", dtype=AMP):
            z = model.encode_adapted(x[i:i + bs])
            l = model.logits_from_neck(z, margin=False)
        lg.append(l.float().cpu()); zz.append(z.float().cpu())
    return torch.cat(lg), torch.cat(zz)


def per_class_prf(preds, y):
    out = {}
    for c in range(NC):
        tp = int(((preds == c) & (y == c)).sum())
        fp = int(((preds == c) & (y != c)).sum())
        fn = int(((preds != c) & (y == c)).sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        out[C[c]] = (p, r, f)
    return out


def main():
    tgt = lib.load_manifest("test", keep_7c=True).reset_index(drop=True)
    yt = tgt["class_idx_7c"].to_numpy()
    val = lib.load_manifest("val", keep_7c=True).reset_index(drop=True)
    yv = val["class_idx_7c"].to_numpy()
    xt = cache(tgt)
    xv = cache(val)
    xv77 = cache(val, das_to=77.0)        # source-val pushed to 77GHz Doppler scale
    print(f"[data] target-418={len(tgt)} source-val={len(val)}")

    model = lib.TimmBackboneV921(config.DEFAULT_BACKBONE, NC, adapter_mode="lora").to(DEVICE).eval()

    conf = np.zeros((NC, NC), dtype=int)          # summed over seeds (true x pred)
    tgt_f1 = {c: [] for c in C}
    src_f1 = {c: [] for c in C}
    flip = {c: [] for c in C}                     # 1 - agreement(clean vs DAS->77) per class
    fcos = {c: [] for c in C}                     # feature cosine clean vs DAS->77 per class

    for s in SEEDS:
        ck = torch.load(REV / "A_V20" / f"seed{s}" / "checkpoints" / "pool_ep100_ema.pt",
                        map_location=DEVICE, weights_only=False)
        model.load_state_dict({k: v.to(DEVICE) for k, v in ck["ema"].items()}, strict=False)

        lt, _ = fwd(model, xt); pt = lt.argmax(1).numpy()
        for t, p in zip(yt, pt):
            conf[t, p] += 1
        for c, (_, _, f) in per_class_prf(pt, yt).items():
            tgt_f1[c].append(f)

        lv, zv = fwd(model, xv); pv = lv.argmax(1).numpy()
        for c, (_, _, f) in per_class_prf(pv, yv).items():
            src_f1[c].append(f)

        lv77, zv77 = fwd(model, xv77); pv77 = lv77.argmax(1).numpy()
        cos = F.cosine_similarity(zv, zv77, dim=1).numpy()
        agree = (pv == pv77).astype(float)
        for ci in range(NC):
            m = yv == ci
            flip[C[ci]].append(float(1 - agree[m].mean()))
            fcos[C[ci]].append(float(cos[m].mean()))

    def mean(d, c): return float(np.mean(d[c]))

    print("\n=== per-class (mean over 3 seeds) ===")
    print(f"{'class':<9}{'tgt77_F1':>9}{'srcval_F1':>10}{'gap(src-tgt)':>13}"
          f"{'flip@77':>9}{'featcos@77':>11}")
    rows = []
    for c in C:
        t = mean(tgt_f1, c); sv = mean(src_f1, c); gap = sv - t
        fl = mean(flip, c); fc = mean(fcos, c)
        rows.append((c, t, sv, gap, fl, fc))
        print(f"{c:<9}{t:>9.3f}{sv:>10.3f}{gap:>13.3f}{fl:>9.3f}{fc:>11.3f}")

    print("\n=== confusion matrix on 77GHz-418 (rows=true, cols=pred, summed 3 seeds) ===")
    print("true\\pred  " + " ".join(f"{c[:4]:>5}" for c in C))
    for i, c in enumerate(C):
        print(f"{c:<10} " + " ".join(f"{conf[i,j]:>5}" for j in range(NC)))

    print("\n=== top off-diagonal confusions (true -> pred, count over 3 seeds) ===")
    pairs = []
    for i in range(NC):
        for j in range(NC):
            if i != j and conf[i, j] > 0:
                pairs.append((conf[i, j], C[i], C[j]))
    for n, ti, tj in sorted(pairs, reverse=True)[:10]:
        print(f"  {ti:>8} -> {tj:<8}  {n}")

    Path("EXPERIMENTSRESULT/CKPT_SELECTION_DIAG/diagnose_improvement.json").write_text(
        json.dumps({"per_class": rows, "confusion": conf.tolist(), "classes": C}, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()
