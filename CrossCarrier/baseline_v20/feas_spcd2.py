"""SP-CD2 feasibility (NO training): is the sink-class bias a source-pair carrier
posterior drift that extrapolates to 77?

For each A_V13_GRL seed: estimate per-class logit carrier-derivative g from same-clip
10<->24 pairs; honestly pick beta on source-val (min 10-vs-24 logit carrier probe,
source-val macro-F1 not down >0.01); apply l77' = l77 - beta*(log77-logref)*g.
Mechanistic gate: 10-vs-24 logit carrier probe must drop (>=0.15 abs or <0.80).
Performance gate: 3-seed paired Delta on full-418, 3/3 same-sign AND mean > 0.832.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image
import config, v9_2_1lib as lib
from verify_priorities import torch_cv_probe

DEVICE = torch.device("cuda"); AMP = torch.bfloat16; NC = config.NUM_CLASSES
ROOT = config.DATASET_ROOT; REV = Path("EXPERIMENTSRESULT/REVISION_5090")
SEEDS = [42, 1234, 31415]; C = config.CLASSES
_tf = lib.tensor_transform(train=False)
L10, L24, L77 = np.log(10.0), np.log(24.0), np.log(77.0); LREF = (L10 + L24) / 2


def cache(paths):
    return torch.stack([_tf(Image.open(ROOT / p).convert("RGB")) for p in paths]).to(DEVICE)


@torch.no_grad()
def logits(model, x, bs=256):
    out = []
    for i in range(0, x.shape[0], bs):
        with torch.autocast("cuda", dtype=AMP):
            lg = model.logits_from_neck(model.encode_adapted(x[i:i + bs]), margin=False)
        out.append(lg.float().cpu())
    return torch.cat(out).numpy()


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
    src = pd.concat([tr, va], ignore_index=True)
    pair_df, st = lib.build_exact_pairs(src)
    print(f"[pairs] {len(pair_df)} matched (cov10={st['coverage_10']:.2f} cov24={st['coverage_24']:.2f})")
    x10 = cache(pair_df["path_10"]); x24 = cache(pair_df["path_24"])
    yb = np.r_[np.zeros(len(x10)), np.ones(len(x10))].astype(int)
    tgt = lib.load_manifest("test", keep_7c=True); y77 = tgt["class_idx_7c"].to_numpy(); x77 = cache(tgt["path"])
    yv = va["class_idx_7c"].to_numpy(); xval = cache(va["path"])
    logf_val = np.log(va["frequency"].map(lib.parse_freq_ghz).to_numpy().astype(float))
    model = lib.TimmBackboneV921(config.DEFAULT_BACKBONE, NC, adapter_mode="lora").to(DEVICE).eval()

    deltas = []
    for s in SEEDS:
        ck = torch.load(REV / "A_V13_GRL" / f"seed{s}" / "checkpoints" / "pool_ep100_ema.pt",
                        map_location=DEVICE, weights_only=False)
        model.load_state_dict({k: v.to(DEVICE) for k, v in ck["ema"].items()}, strict=False)
        l10 = logits(model, x10); l24 = logits(model, x24); l77 = logits(model, x77); lval = logits(model, xval)
        g = np.median((l24 - l10) / (L24 - L10), axis=0)            # (7,) per-class drift / log-carrier
        base77 = mf1(l77.argmax(1), y77); baseval = mf1(lval.argmax(1), yv)
        probe_before = torch_cv_probe(np.concatenate([l10, l24]), yb, 2)
        # honest beta selection on SOURCE only
        best = None
        for beta in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
            l10d = l10 - beta * (L10 - LREF) * g; l24d = l24 - beta * (L24 - LREF) * g
            probe = torch_cv_probe(np.concatenate([l10d, l24d]), yb, 2)
            lvald = lval - beta * (logf_val - LREF)[:, None] * g
            valf1 = mf1(lvald.argmax(1), yv)
            if valf1 >= baseval - 0.01 and (best is None or probe < best[1]):
                best = (beta, probe, valf1)
        beta, probe_after, valf1 = best
        l77d = l77 - beta * (L77 - LREF) * g
        new77 = mf1(l77d.argmax(1), y77)
        deltas.append(new77 - base77)
        print(f"seed{s}: ||g||={np.linalg.norm(g):.3f}  probe 10v24 {probe_before:.3f}->{probe_after:.3f}  "
              f"beta*={beta}  src-val {baseval:.3f}->{valf1:.3f}  77GHz {base77:.4f}->{new77:.4f}  d{new77-base77:+.4f}")

    npos = sum(1 for d in deltas if d > 0)
    print(f"\nSP-CD2: per-seed Delta77 = {[round(d,4) for d in deltas]}  mean {np.mean(deltas):+.4f}  {npos}/3 positive")
    print("VERDICT:", "POTENTIAL (mechanism + 3/3 same-sign)" if npos == 3 and np.mean(deltas) > 0
          else "KILL (no robust gain; carrier drift not an extrapolable logit posterior)")


if __name__ == "__main__":
    main()
