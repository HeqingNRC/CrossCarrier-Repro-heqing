"""Which mechanism produces the ensemble gain? Decompose, don't hand-wave.
- Is it decorrelated-error variance reduction (majority recovers split samples)?
- Does it = majority vote (pure voting) or does logit-confidence add more?
- Do the seeds disagree MOST on the high-realism-gap classes (Sit/Towards)?  -> links
  the variance to the underdetermined carrier->sensor boundary.
"""
from __future__ import annotations
import numpy as np, torch
from pathlib import Path
from PIL import Image
import config, v9_2_1lib as lib
DEVICE = torch.device("cuda"); AMP = torch.bfloat16; NC = config.NUM_CLASSES
ROOT = config.DATASET_ROOT; REV = Path("EXPERIMENTSRESULT/REVISION_5090"); SEEDS = [42, 1234, 31415]
C = config.CLASSES; _tf = lib.tensor_transform(train=False)
REAL_EXCESS = {"Sit": 0.40, "Towards": 0.21, "Bend": 0.06, "Away": 0.05, "Kneel": 0.05, "Pick": 0.04, "SStep": 0.04}


def cache(df):
    return torch.stack([_tf(Image.open(ROOT / p).convert("RGB")) for p in df["path"]]).to(DEVICE)


@torch.no_grad()
def logits(model, x, bs=256):
    o = []
    for i in range(0, x.shape[0], bs):
        with torch.autocast("cuda", dtype=AMP):
            o.append(model.logits_from_neck(model.encode_adapted(x[i:i + bs]), margin=False).float().cpu())
    return torch.cat(o).numpy()


def macro_f1(pred, y):
    f = []
    for c in range(NC):
        tp = int(((pred == c) & (y == c)).sum()); fp = int(((pred == c) & (y != c)).sum()); fn = int(((pred != c) & (y == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
        f.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f))


def main():
    full = lib.load_manifest("test", keep_7c=True).reset_index(drop=True)
    y = full["class_idx_7c"].to_numpy(); x = cache(full); N = len(y)
    model = lib.TimmBackboneV921(config.DEFAULT_BACKBONE, NC, adapter_mode="lora").to(DEVICE).eval()
    P, Lg = [], []
    for s in SEEDS:
        ck = torch.load(REV / "A_V13_GRL" / f"seed{s}" / "checkpoints" / "pool_ep100_ema.pt",
                        map_location=DEVICE, weights_only=False)
        model.load_state_dict({k: v.to(DEVICE) for k, v in ck["ema"].items()}, strict=False)
        lg = logits(model, x); Lg.append(lg); P.append(lg.argmax(1))
    P = np.stack(P)                       # (3,N)
    ens = np.mean(Lg, 0).argmax(1)
    # majority vote (ties -> lowest class index, rare)
    maj = np.array([np.bincount(P[:, i], minlength=NC).argmax() for i in range(N)])

    ncorr = (P == y).sum(0)               # per-sample #seeds correct (0..3)
    seed_acc = np.mean([(P[i] == y).mean() for i in range(3)])
    print(f"single-seed acc mean={seed_acc:.4f}  majority-vote acc={ (maj==y).mean():.4f}  "
          f"logit-ens acc={ (ens==y).mean():.4f}")
    print(f"single-seed macroF1 mean={np.mean([macro_f1(P[i],y) for i in range(3)]):.4f}  "
          f"maj-vote F1={macro_f1(maj,y):.4f}  logit-ens F1={macro_f1(ens,y):.4f}")

    allagree = ((P[0] == P[1]) & (P[1] == P[2])).mean()
    print(f"\nagreement: all-3-agree on {allagree:.1%} of samples")
    print("\nensemble correctness vs #seeds-correct (where the gain lives):")
    for k in range(4):
        m = ncorr == k
        if m.sum():
            print(f"  {k}/3 seeds correct: n={int(m.sum())}  ens correct here={(ens[m]==y[m]).mean():.2f}")
    # net gain decomposition
    ens_right_seed_wrong = int(((ens == y) & (ncorr <= 1)).sum())   # ens rescues a minority/lost sample
    ens_wrong_seed_right = int(((ens != y) & (ncorr >= 2)).sum())   # ens breaks a majority-right sample
    print(f"\nNET: ens rescues (ens right, <=1 seed right) = {ens_right_seed_wrong}; "
          f"ens breaks (ens wrong, >=2 seeds right) = {ens_wrong_seed_right}")

    print("\nper-class DISAGREEMENT rate (frac of class samples where the 3 seeds are NOT unanimous) vs realism excess:")
    print(f"{'class':9s} {'disagree%':>10} {'realism_excess':>15}")
    rows = []
    for ci, cc in enumerate(C):
        m = y == ci
        dis = np.mean(P[0, m] != P[1, m]) if m.sum() else 0
        dis = np.mean([(len(set(P[:, i])) > 1) for i in np.where(m)[0]]) if m.sum() else 0
        rows.append((cc, dis, REAL_EXCESS.get(cc, 0)))
        print(f"{cc:9s} {dis:>10.1%} {REAL_EXCESS.get(cc,0):>15.2f}")
    dr = np.array([r[1] for r in rows]); ex = np.array([r[2] for r in rows])
    print(f"\ncorr(disagreement, realism_excess) over classes = {np.corrcoef(dr, ex)[0,1]:+.2f}")


if __name__ == "__main__":
    main()
