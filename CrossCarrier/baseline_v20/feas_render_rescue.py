"""FAIR re-test (no training) of the rendering family that was killed on the saturated
0.999 probe. Metric = per-class centroid DISTANCE (magnitude) real77-vs-DAS77 in the
TRAINED z_cls (where Sit excess is 0.356), with the within-real77 floor as reference.
A transform that LOWERS Sit/Towards excess is a LEAD (collect for the user; no training).

Transforms applied to the DAS-source->77 views (training synthetic): identity (baseline),
CR-DAS2 Doppler-axis blur (sigma) and sharpen (unsharp), LMQ log-Doppler vertical warp.
"""
from __future__ import annotations
import numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from PIL import Image
import config, v9_2_1lib as lib
from verify_priorities import torch_cv_probe
np.random.seed(0)
DEVICE = torch.device("cuda"); AMP = torch.bfloat16; NC = config.NUM_CLASSES
ROOT = config.DATASET_ROOT; C = config.CLASSES; SI = {c: i for i, c in enumerate(C)}
REV = Path("EXPERIMENTSRESULT/REVISION_5090")


def t01(pil):
    im = pil.convert("RGB").resize((224, 224), Image.BILINEAR)
    return torch.from_numpy(np.asarray(im, np.float32).transpose(2, 0, 1) / 255.0)


def vblur(x, sigma):
    if sigma <= 0:
        return x
    k = max(3, int(6 * sigma) | 1); r = k // 2
    xs = torch.arange(k, device=x.device, dtype=torch.float32) - r
    w = torch.exp(-(xs ** 2) / (2 * sigma ** 2)); w = (w / w.sum()).view(1, 1, k, 1).repeat(x.shape[1], 1, 1, 1)
    return F.conv2d(x, w, padding=(r, 0), groups=x.shape[1])


def logwarp(x, a):
    # remap vertical (Doppler) axis log-style around center; a=strength (0=identity)
    B, Cc, H, W = x.shape
    v = torch.linspace(-1, 1, H, device=x.device)
    vin = torch.sign(v) * (torch.exp(a * v.abs()) - 1) / (np.exp(a) - 1 + 1e-6) if a > 0 else v
    gy = vin.view(1, H, 1).expand(B, H, W)
    gx = torch.linspace(-1, 1, W, device=x.device).view(1, 1, W).expand(B, H, W)
    grid = torch.stack([gx, gy], dim=-1)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)


def transform(x, name):
    if name == "identity":
        return x
    if name.startswith("blur"):
        return vblur(x, float(name[4:]))
    if name.startswith("sharp"):
        amt = float(name[5:]); return (x + amt * (x - vblur(x, 1.0))).clamp(0, 1)
    if name.startswith("logwarp"):
        return logwarp(x, float(name[7:]))
    raise ValueError(name)


def std(t):
    return (t - t.mean((2, 3), keepdim=True)) / (t.std((2, 3), keepdim=True) + 1e-6)


@torch.no_grad()
def zcls(model, imgs01, name, bs=128):
    out = []
    for i in range(0, imgs01.shape[0], bs):
        xb = transform(imgs01[i:i + bs].to(DEVICE), name)
        with torch.autocast("cuda", dtype=AMP):
            out.append(model.encode_adapted(std(xb)).float().cpu())
    z = torch.cat(out).numpy(); return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)


def excess(zd, yd, zr, yr, floor):
    out = {}
    for cc in ["Sit", "Towards", "Bend"]:
        ci = SI[cc]; a = zd[yd == ci]; b = zr[yr == ci]
        if len(a) >= 4 and len(b) >= 4:
            ca = a.mean(0); ca /= np.linalg.norm(ca) + 1e-8; cb = b.mean(0); cb /= np.linalg.norm(cb) + 1e-8
            out[cc] = float(1 - ca @ cb) - floor[cc]
    return out


def main():
    va = lib.load_manifest("val", keep_7c=True); tgt = lib.load_manifest("test", keep_7c=True)
    yv = va["class_idx_7c"].to_numpy(); y77 = tgt["class_idx_7c"].to_numpy()
    real = torch.stack([t01(Image.open(ROOT / p)) for p in tgt["path"]])
    das = torch.stack([t01(lib.das_deterministic(Image.open(ROOT / p).convert("RGB"),
                       lib.parse_freq_ghz(f), 77.0)) for p, f in zip(va["path"], va["frequency"])])
    model = lib.TimmBackboneV921(config.DEFAULT_BACKBONE, NC, adapter_mode="lora").to(DEVICE).eval()
    ck = torch.load(REV / "A_V13_GRL" / "seed42" / "checkpoints" / "pool_ep100_ema.pt",
                    map_location=DEVICE, weights_only=False)
    model.load_state_dict({k: v.to(DEVICE) for k, v in ck["ema"].items()}, strict=False)

    zr = zcls(model, real, "identity")
    # within-real77 floor per class
    floor = {}
    for cc in ["Sit", "Towards", "Bend"]:
        b = zr[y77 == SI[cc]]; h = np.random.permutation(len(b)); half = len(b) // 2
        ca = b[h[:half]].mean(0); ca /= np.linalg.norm(ca) + 1e-8
        cb = b[h[half:]].mean(0); cb /= np.linalg.norm(cb) + 1e-8
        floor[cc] = float(1 - ca @ cb)
    print(f"trained-z_cls real77-vs-DAS77 EXCESS (over within-real floor); lower=closer. floor={ {k:round(v,3) for k,v in floor.items()} }")
    print(f"{'transform':10s} {'Sit':>7} {'Towards':>8} {'Bend':>7}")
    for T in ["identity", "blur0.5", "blur1.0", "blur2.0", "sharp0.5", "sharp1.0", "logwarp0.5", "logwarp1.0"]:
        zd = zcls(model, das, T)
        ex = excess(zd, yv, zr, y77, floor)
        print(f"{T:10s} {ex.get('Sit',float('nan')):>7.3f} {ex.get('Towards',float('nan')):>8.3f} {ex.get('Bend',float('nan')):>7.3f}")
    print("\nLEAD iff some transform lowers Sit AND Towards excess >=20% vs identity. Else fair negative.")


if __name__ == "__main__":
    main()
