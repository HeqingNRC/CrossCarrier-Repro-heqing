"""DOLI kill-test: fixed inference-time averaging of A_V13_GRL logits over a
pre-registered DAS carrier orbit (7 log-spaced anchors in [12,95] GHz). Group/
nuisance marginalization -- NOT margin (dead-end C), NOT drift extrapolation (E).

Mechanistic (source-only): on the 361 10<->24 pairs, the 10-vs-24 logit carrier
probe must drop >=0.15 (or <0.53), with source-val macro-F1 down <=0.01.
Performance: full-418, 3 seeds, DOLI vs baseline -> 3/3 positive AND mean > 0.832.
"""
from __future__ import annotations
import numpy as np, pandas as pd, torch, math
from pathlib import Path
from PIL import Image
import config, v9_2_1lib as lib
from verify_priorities import torch_cv_probe

DEVICE = torch.device("cuda"); AMP = torch.bfloat16; NC = config.NUM_CLASSES
ROOT = config.DATASET_ROOT; REV = Path("EXPERIMENTSRESULT/REVISION_5090")
SEEDS = [42, 1234, 31415]; C = config.CLASSES
ANCHORS = [math.exp(t) for t in np.linspace(math.log(12), math.log(95), 7)]


def base01(paths):
    out = []
    for p in paths:
        im = Image.open(ROOT / p).convert("RGB").resize((224, 224), Image.BILINEAR)
        out.append(torch.from_numpy(np.asarray(im, np.float32).transpose(2, 0, 1) / 255.0))
    return torch.stack(out).to(DEVICE)


def std(t):
    return (t - t.mean((2, 3), keepdim=True)) / (t.std((2, 3), keepdim=True) + 1e-6)


@torch.no_grad()
def logits_raw(model, b01, bs=256):
    out = []
    for i in range(0, b01.shape[0], bs):
        with torch.autocast("cuda", dtype=AMP):
            out.append(model.logits_from_neck(model.encode_adapted(std(b01[i:i + bs])), margin=False).float().cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def logits_doli(model, b01, fsrc, bs=256):
    acc = np.zeros((b01.shape[0], NC), np.float32)
    for fa in ANCHORS:
        sc = (fa / fsrc).to(DEVICE)
        for i in range(0, b01.shape[0], bs):
            with torch.autocast("cuda", dtype=AMP):
                v = std(lib.gpu_das(b01[i:i + bs], sc[i:i + bs]))
                lg = model.logits_from_neck(model.encode_adapted(v), margin=False).float().cpu().numpy()
            acc[i:i + bs] += lg
    return acc / len(ANCHORS)


def mf1(pred, y):
    f = []
    for c in range(NC):
        tp = int(((pred == c) & (y == c)).sum()); fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
        f.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f))


def main():
    tr = lib.load_manifest("train", keep_7c=True); va = lib.load_manifest("val", keep_7c=True)
    pair_df, _ = lib.build_exact_pairs(pd.concat([tr, va], ignore_index=True))
    yb = np.r_[np.zeros(len(pair_df)), np.ones(len(pair_df))].astype(int)
    b10 = base01(pair_df["path_10"]); f10 = torch.full((len(pair_df),), 10.0)
    b24 = base01(pair_df["path_24"]); f24 = torch.full((len(pair_df),), 24.0)
    tgt = lib.load_manifest("test", keep_7c=True); y77 = tgt["class_idx_7c"].to_numpy()
    b77 = base01(tgt["path"]); f77 = torch.full((len(tgt),), 77.0)
    yv = va["class_idx_7c"].to_numpy(); bval = base01(va["path"])
    fval = torch.tensor(va["frequency"].map(lib.parse_freq_ghz).to_numpy().astype("float32"))
    model = lib.TimmBackboneV921(config.DEFAULT_BACKBONE, NC, adapter_mode="lora").to(DEVICE).eval()

    deltas = []
    for s in SEEDS:
        ck = torch.load(REV / "A_V13_GRL" / f"seed{s}" / "checkpoints" / "pool_ep100_ema.pt",
                        map_location=DEVICE, weights_only=False)
        model.load_state_dict({k: v.to(DEVICE) for k, v in ck["ema"].items()}, strict=False)
        # mechanistic (source pairs)
        p_base = torch_cv_probe(np.concatenate([logits_raw(model, b10), logits_raw(model, b24)]), yb, 2)
        p_doli = torch_cv_probe(np.concatenate([logits_doli(model, b10, f10), logits_doli(model, b24, f24)]), yb, 2)
        # source-val macro-F1
        vbase = mf1(logits_raw(model, bval).argmax(1), yv)
        vdoli = mf1(logits_doli(model, bval, fval).argmax(1), yv)
        # performance (77)
        base77 = mf1(logits_raw(model, b77).argmax(1), y77)
        doli77 = mf1(logits_doli(model, b77, f77).argmax(1), y77)
        deltas.append(doli77 - base77)
        print(f"seed{s}: 10v24 logit probe {p_base:.3f}->{p_doli:.3f}  src-val {vbase:.3f}->{vdoli:.3f}  "
              f"77GHz {base77:.4f}->{doli77:.4f}  d{doli77-base77:+.4f}")

    npos = sum(1 for d in deltas if d > 0)
    print(f"\nDOLI: per-seed Delta77 = {[round(d,4) for d in deltas]}  mean {np.mean(deltas):+.4f}  {npos}/3 positive")
    print("VERDICT DOLI:", "POTENTIAL (3/3 positive, mean>base)" if npos == 3 and np.mean(deltas) > 0
          else "KILL (orbit logit averaging does not robustly help 77)")


if __name__ == "__main__":
    main()
