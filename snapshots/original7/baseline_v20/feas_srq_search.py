"""SRQ constructive SEARCH (find the best intensity canonicalization, not kill one).
Metric = per-class centroid DISTANCE (magnitude) real77-vs-DAS77 in the carrier-naive
oracle (NOT the saturated linear probe), subject to preserving the source 7-way class
probe. Search a family of monotone intensity transforms incl. histogram-matching to a
source reference (the proper 'remove monotone sensor response'). Find any that shrinks
the Sit/Towards gap while keeping source class info -> escalate that one to training.
"""
from __future__ import annotations
import numpy as np, torch
from pathlib import Path
from PIL import Image
import config, v9_2_1lib as lib
from verify_priorities import torch_cv_probe
np.random.seed(0)
DEVICE = torch.device("cuda"); AMP = torch.bfloat16; NC = config.NUM_CLASSES
ROOT = config.DATASET_ROOT; C = config.CLASSES
try:
    import matplotlib.cm as cm
    _JET = torch.tensor(cm.get_cmap("jet")(np.linspace(0, 1, 256))[:, :3], dtype=torch.float32, device=DEVICE)
except Exception:
    xs = np.linspace(0, 1, 256)
    _JET = torch.tensor(np.stack([np.clip(1.5 - np.abs(4*xs-3), 0, 1), np.clip(1.5-np.abs(4*xs-2), 0, 1),
                                  np.clip(1.5-np.abs(4*xs-1), 0, 1)], 1), dtype=torch.float32, device=DEVICE)


def t01(pil):
    im = pil.convert("RGB").resize((224, 224), Image.BILINEAR)
    return torch.from_numpy(np.asarray(im, np.float32).transpose(2, 0, 1) / 255.0)


def to_scalar(rgb):  # (B,3,H,W)->(B,H,W)
    rgb = rgb.to(DEVICE); B, _, H, W = rgb.shape
    s = torch.cdist(rgb.permute(0, 2, 3, 1).reshape(-1, 3), _JET).argmin(1).float() / 255.0
    return s.view(B, H, W)


def src_ref_cdf(scal):  # pooled foreground sorted values -> reference quantile values
    fg = scal[scal >= 0.10].cpu().numpy()
    return np.sort(np.random.choice(fg, size=min(200000, len(fg)), replace=False))


def apply_T(scal, name, ref=None):
    out = scal.clone()
    for b in range(scal.shape[0]):
        flat = out[b].reshape(-1); fg = flat >= 0.10
        v = flat[fg]
        if v.numel() < 2:
            continue
        if name == "identity":
            pass
        elif name == "gamma0.5":
            v = v ** 0.5
        elif name == "gamma2.0":
            v = v ** 2.0
        elif name == "histeq":
            v = torch.argsort(torch.argsort(v)).float() / (v.numel() - 1)
        elif name == "histmatch":
            q = (torch.argsort(torch.argsort(v)).float() / (v.numel() - 1)).cpu().numpy()
            v = torch.tensor(np.interp(q, np.linspace(0, 1, len(ref)), ref), device=DEVICE, dtype=torch.float32)
        flat[fg] = v
        out[b] = flat.view(scal.shape[1], scal.shape[2])
    t = out.unsqueeze(1).repeat(1, 3, 1, 1)
    return (t - t.mean((2, 3), keepdim=True)) / (t.std((2, 3), keepdim=True) + 1e-6)


@torch.no_grad()
def oracle(model, tens, bs=128):
    o = []
    for i in range(0, tens.shape[0], bs):
        with torch.autocast("cuda", dtype=AMP):
            o.append(model.oracle_encoder(tens[i:i + bs].to(DEVICE)).float().cpu())
    z = torch.cat(o).numpy(); return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)


def cdist_cls(zr, yr, zd, yd, cc):
    a = zd[yd == cc]; b = zr[yr == cc]
    if len(a) < 4 or len(b) < 4:
        return None
    ca = a.mean(0); ca /= np.linalg.norm(ca) + 1e-8; cb = b.mean(0); cb /= np.linalg.norm(cb) + 1e-8
    return float(1 - ca @ cb)


def main():
    va = lib.load_manifest("val", keep_7c=True); tgt = lib.load_manifest("test", keep_7c=True)
    yv = va["class_idx_7c"].to_numpy(); y77 = tgt["class_idx_7c"].to_numpy()
    real = torch.stack([t01(Image.open(ROOT / p)) for p in tgt["path"]])
    das = torch.stack([t01(lib.das_deterministic(Image.open(ROOT / p).convert("RGB"),
                       lib.parse_freq_ghz(f), 77.0)) for p, f in zip(va["path"], va["frequency"])])
    val = torch.stack([t01(Image.open(ROOT / p)) for p in va["path"]])
    s_real, s_das, s_val = to_scalar(real), to_scalar(das), to_scalar(val)
    ref = src_ref_cdf(s_val)
    model = lib.TimmBackboneV921(config.DEFAULT_BACKBONE, NC, adapter_mode="lora").to(DEVICE).eval()
    SI = {c: i for i, c in enumerate(C)}

    print(f"{'transform':10s} {'src7way':>8} {'Sit':>7} {'Towards':>8} {'Bend':>7} {'meanWeak':>9}")
    base = {}
    for T in ["identity", "gamma0.5", "gamma2.0", "histeq", "histmatch"]:
        zr = oracle(model, apply_T(s_real, T, ref)); zd = oracle(model, apply_T(s_das, T, ref))
        zv = oracle(model, apply_T(s_val, T, ref))
        src = torch_cv_probe(zv, yv, NC)
        d = {cc: cdist_cls(zr, y77, zd, yv, SI[cc]) for cc in ["Sit", "Towards", "Bend"]}
        mw = np.mean([d[c] for c in d])
        if T == "identity":
            base = dict(d); base["src"] = src
        print(f"{T:10s} {src:>8.3f} {d['Sit']:>7.3f} {d['Towards']:>8.3f} {d['Bend']:>7.3f} {mw:>9.3f}")

    print("\nReading: vs identity, a GOOD transform LOWERS Sit/Towards distance while keeping src7way ~constant.")
    print("If histmatch/gamma lowers weak-class distance >=20% with src drop <=0.03 -> escalate to seed42 training.")


if __name__ == "__main__":
    main()
