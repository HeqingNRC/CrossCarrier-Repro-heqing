"""SP-CD2 FAIR re-exam: was it selection (not mechanism) that failed? Report the
ORACLE best-beta upper bound (target-peeking, NOT a claimable result) + cross-seed
consistency, for two references. If even the oracle-best beta gives only noise / no
shared-beta 3/3 gain, SP-CD2 is truly dead; if a shared beta gives a real consistent
gain, the mechanism has signal and only honest selection is missing (a LEAD to collect).
"""
from __future__ import annotations
import numpy as np, pandas as pd, torch
from pathlib import Path
from PIL import Image
import config, v9_2_1lib as lib

DEVICE = torch.device("cuda"); AMP = torch.bfloat16; NC = config.NUM_CLASSES
ROOT = config.DATASET_ROOT; REV = Path("EXPERIMENTSRESULT/REVISION_5090"); SEEDS = [42, 1234, 31415]
_tf = lib.tensor_transform(train=False)
L10, L24, L77 = np.log(10.0), np.log(24.0), np.log(77.0)
BETAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]


def cache(paths):
    return torch.stack([_tf(Image.open(ROOT / p).convert("RGB")) for p in paths]).to(DEVICE)


@torch.no_grad()
def logits(model, x, bs=256):
    o = []
    for i in range(0, x.shape[0], bs):
        with torch.autocast("cuda", dtype=AMP):
            o.append(model.logits_from_neck(model.encode_adapted(x[i:i + bs]), margin=False).float().cpu())
    return torch.cat(o).numpy()


def mf1(pred, y):
    f = []
    for c in range(NC):
        tp = int(((pred == c) & (y == c)).sum()); fp = int(((pred == c) & (y != c)).sum()); fn = int(((pred != c) & (y == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
        f.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f))


def main():
    tr = lib.load_manifest("train", keep_7c=True); va = lib.load_manifest("val", keep_7c=True)
    pair_df, _ = lib.build_exact_pairs(pd.concat([tr, va], ignore_index=True))
    x10 = cache(pair_df["path_10"]); x24 = cache(pair_df["path_24"])
    tgt = lib.load_manifest("test", keep_7c=True); y77 = tgt["class_idx_7c"].to_numpy(); x77 = cache(tgt["path"])
    model = lib.TimmBackboneV921(config.DEFAULT_BACKBONE, NC, adapter_mode="lora").to(DEVICE).eval()

    for ref_name, LREF in [("ref=mean(10,24)", (L10 + L24) / 2), ("ref=log24", L24)]:
        print(f"\n==== {ref_name} : 77GHz macro-F1 vs beta (per seed) ====")
        print("beta:    " + "  ".join(f"{b:>5}" for b in BETAS))
        tab = {}
        for s in SEEDS:
            ck = torch.load(REV / "A_V13_GRL" / f"seed{s}" / "checkpoints" / "pool_ep100_ema.pt",
                            map_location=DEVICE, weights_only=False)
            model.load_state_dict({k: v.to(DEVICE) for k, v in ck["ema"].items()}, strict=False)
            l10 = logits(model, x10); l24 = logits(model, x24); l77 = logits(model, x77)
            g = np.median((l24 - l10) / (L24 - L10), axis=0)
            row = [mf1((l77 - b * (L77 - LREF) * g).argmax(1), y77) for b in BETAS]
            tab[s] = row
            print(f"s{s:<6} " + "  ".join(f"{v:.3f}" for v in row))
        base = [tab[s][0] for s in SEEDS]
        # best SHARED beta (excluding 0): does any single beta beat base on 3/3?
        best_shared = None
        for j, b in enumerate(BETAS):
            if b == 0:
                continue
            deltas = [tab[s][j] - tab[s][0] for s in SEEDS]
            if all(d > 0 for d in deltas):
                mean_new = np.mean([tab[s][j] for s in SEEDS])
                if best_shared is None or mean_new > best_shared[1]:
                    best_shared = (b, mean_new, deltas)
        oracle_best = np.mean([max(tab[s]) for s in SEEDS])
        print(f"base 3-seed mean={np.mean(base):.4f}; ORACLE per-seed-best mean={oracle_best:.4f}")
        if best_shared:
            print(f"BEST SHARED beta={best_shared[0]} -> 3/3 positive, mean={best_shared[1]:.4f}, deltas={[round(d,4) for d in best_shared[2]]}  <- LEAD")
        else:
            print("NO single shared beta is 3/3 positive -> SP-CD2 dead even with oracle selection")


if __name__ == "__main__":
    main()
