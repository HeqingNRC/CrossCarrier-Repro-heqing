"""Paired stratified bootstrap 95% CI for continuous vs discrete carrier adversary.

Compares full ACR with the CONTINUOUS log-carrier regressor (A_V13_GRL) against full ACR
with the DISCRETE 3-bin carrier adversary (SUPP_ABLATION/ACR_discrete). Both: final-EMA
(ep100), full-418 77 GHz, seeds 42/1234/31415. "single" macro-F1 on a resample = mean over
the 3 seeds. Resampling is paired (same clip indices both sides) and class-stratified.
"""
from __future__ import annotations
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
B = 10000
RNG = 0
_tf = lib.tensor_transform(train=False)


def cache(df):
    return torch.stack([_tf(Image.open(ROOT / p).convert("RGB")) for p in df["path"]]).to(DEVICE)


def resolve(seed_dir: Path):
    p = seed_dir / "checkpoints" / "pool_ep100_ema.pt"
    if p.exists():
        return p
    return seed_dir / "epoch_ckpts" / "ep100.pt"


@torch.no_grad()
def get_logits(model, x, bs=256):
    out = []
    for i in range(0, x.shape[0], bs):
        with torch.autocast("cuda", dtype=AMP):
            z = model.encode_adapted(x[i:i + bs])
            out.append(model.logits_from_neck(z, margin=False).float().cpu())
    return torch.cat(out).numpy()


def macro_f1(preds, y, idx):
    p, t = preds[idx], y[idx]
    fs = []
    for c in range(NC):
        tp = int(((p == c) & (t == c)).sum()); fp = int(((p == c) & (t != c)).sum()); fn = int(((p != c) & (t == c)).sum())
        pr = tp / (tp + fp) if (tp + fp) else 0.0; rc = tp / (tp + fn) if (tp + fn) else 0.0
        fs.append(2 * pr * rc / (pr + rc) if (pr + rc) else 0.0)
    return float(np.mean(fs))


def single_f1(L, y, idx):
    preds = L.argmax(2)
    return float(np.mean([macro_f1(preds[s], y, idx) for s in range(L.shape[0])]))


def load_cfg(base):
    model = lib.TimmBackboneV921(config.DEFAULT_BACKBONE, NC, adapter_mode="lora").to(DEVICE).eval()
    Ls = []
    for s in SEEDS:
        ck = torch.load(resolve(base / f"seed{s}"), map_location=DEVICE, weights_only=False)
        model.load_state_dict({k: v.to(DEVICE) for k, v in ck["ema"].items()}, strict=False)
        Ls.append(get_logits(model, X))
    return np.stack(Ls, 0)


full = lib.load_manifest("test", keep_7c=True).reset_index(drop=True)
y = full["class_idx_7c"].to_numpy()
X = cache(full)
print(f"[data] full-418={len(full)}")

L_cont = load_cfg(REV / "A_V13_GRL")
L_disc = load_cfg(REV / "SUPP_ABLATION" / "ACR_discrete")
print(f"continuous single mean F1 = {single_f1(L_cont, y, np.arange(len(y))):.4f}")
print(f"discrete   single mean F1 = {single_f1(L_disc, y, np.arange(len(y))):.4f}")

rng = np.random.default_rng(RNG)
by_class = [np.where(y == c)[0] for c in range(NC)]
full_idx = np.arange(len(y))
obs = single_f1(L_cont, y, full_idx) - single_f1(L_disc, y, full_idx)
deltas = np.empty(B)
for b in range(B):
    idx = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in by_class if len(ix)])
    deltas[b] = single_f1(L_cont, y, idx) - single_f1(L_disc, y, idx)
lo, hi = np.percentile(deltas, [2.5, 97.5])
print(f"\ncontinuous - discrete (full ACR): observed = {obs*100:+.2f} pp")
print(f"95% CI = [{lo*100:+.2f}, {hi*100:+.2f}] pp   P(>0) = {(deltas>0).mean():.3f}   B={B}")
