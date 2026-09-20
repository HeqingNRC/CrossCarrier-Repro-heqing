"""SRQ kill-test: is part of the real77-vs-DAS77 gap a SENSOR RESPONSE (jet colormap /
dynamic range), removable by a class-agnostic intensity canonicalization?

SRQ(image) = jet RGB -> scalar (nearest-LUT) -> per-image rank/quantile normalize
(foreground -> uniform CDF, background -> 0) -> triplicate -> standardize.
Measured in the carrier-naive frozen DINOv3 oracle.

PASS iff: oracle real-vs-DAS77 disc 0.984-baseline -> <0.93 ; Sit & Towards centroid
excess each drop >=25% ; source 7-way class probe drops <=0.03 vs raw (action info kept).
"""
from __future__ import annotations
import numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from PIL import Image
import config, v9_2_1lib as lib
from verify_priorities import torch_cv_probe

DEVICE = torch.device("cuda"); AMP = torch.bfloat16; NC = config.NUM_CLASSES
ROOT = config.DATASET_ROOT; C = config.CLASSES; np.random.seed(0)

# jet LUT (256,3) in [0,1]
try:
    import matplotlib.cm as cm
    _JET = torch.tensor(cm.get_cmap("jet")(np.linspace(0, 1, 256))[:, :3], dtype=torch.float32, device=DEVICE)
except Exception:
    xs = np.linspace(0, 1, 256)
    r = np.clip(1.5 - np.abs(4 * xs - 3), 0, 1); g = np.clip(1.5 - np.abs(4 * xs - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * xs - 1), 0, 1)
    _JET = torch.tensor(np.stack([r, g, b], 1), dtype=torch.float32, device=DEVICE)


def to_tensor01(pil):
    im = pil.convert("RGB").resize((224, 224), Image.BILINEAR)
    return torch.from_numpy(np.asarray(im, np.float32).transpose(2, 0, 1) / 255.0)


def jet_to_scalar(rgb):  # rgb (B,3,H,W) -> scalar (B,H,W) in [0,1]
    B, _, H, W = rgb.shape
    px = rgb.permute(0, 2, 3, 1).reshape(-1, 3)
    d = torch.cdist(px, _JET)              # (P,256)
    s = d.argmin(1).float() / 255.0
    return s.view(B, H, W)


def srq_batch(rgb):  # rgb (B,3,H,W) in [0,1] -> standardized (B,3,H,W)
    rgb = rgb.to(DEVICE)
    B, _, H, W = rgb.shape
    s = jet_to_scalar(rgb)                  # (B,H,W)
    out = torch.zeros_like(s)
    for b in range(B):
        flat = s[b].reshape(-1); fg = flat >= 0.10
        if fg.sum() > 1:
            r = torch.argsort(torch.argsort(flat[fg])).float()
            out[b].reshape(-1)[fg] = r / (r.max() + 1e-6)
    t = out.unsqueeze(1).repeat(1, 3, 1, 1)
    return (t - t.mean((2, 3), keepdim=True)) / (t.std((2, 3), keepdim=True) + 1e-6)


@torch.no_grad()
def oracle(model, tens, bs=128):
    out = []
    for i in range(0, tens.shape[0], bs):
        with torch.autocast("cuda", dtype=AMP):
            out.append(model.oracle_encoder(tens[i:i + bs].to(DEVICE)).float().cpu())
    z = torch.cat(out).numpy()
    return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)


def excess(zsyn, ysyn, zreal, yreal):
    out = {}
    for ci, cc in enumerate(C):
        a = zsyn[ysyn == ci]; b = zreal[yreal == ci]
        if len(a) >= 4 and len(b) >= 4:
            ca = a.mean(0); ca /= np.linalg.norm(ca) + 1e-8
            cb = b.mean(0); cb /= np.linalg.norm(cb) + 1e-8
            out[cc] = float(1 - ca @ cb)
    return out


def std_raw(tens):
    return (tens - tens.mean((2, 3), keepdim=True)) / (tens.std((2, 3), keepdim=True) + 1e-6)


def main():
    va = lib.load_manifest("val", keep_7c=True); tgt = lib.load_manifest("test", keep_7c=True)
    yv = va["class_idx_7c"].to_numpy(); y77 = tgt["class_idx_7c"].to_numpy()
    real77 = torch.stack([to_tensor01(Image.open(ROOT / p)) for p in tgt["path"]])
    das77 = torch.stack([to_tensor01(lib.das_deterministic(Image.open(ROOT / p).convert("RGB"),
                          lib.parse_freq_ghz(f), 77.0)) for p, f in zip(va["path"], va["frequency"])])
    valraw = torch.stack([to_tensor01(Image.open(ROOT / p)) for p in va["path"]])
    model = lib.TimmBackboneV921(config.DEFAULT_BACKBONE, NC, adapter_mode="lora").to(DEVICE).eval()

    # baseline (raw, oracle)
    zr = oracle(model, std_raw(real77)); zd = oracle(model, std_raw(das77))
    disc0 = torch_cv_probe(np.concatenate([zr, zd]),
                           np.r_[np.zeros(len(zr)), np.ones(len(zd))].astype(int), 2)
    ex0 = excess(zd, yv, zr, y77)
    src0 = torch_cv_probe(oracle(model, std_raw(valraw)), yv, NC)
    # SRQ
    zr_s = oracle(model, srq_batch(real77)); zd_s = oracle(model, srq_batch(das77))
    disc1 = torch_cv_probe(np.concatenate([zr_s, zd_s]),
                           np.r_[np.zeros(len(zr_s)), np.ones(len(zd_s))].astype(int), 2)
    ex1 = excess(zd_s, yv, zr_s, y77)
    src1 = torch_cv_probe(oracle(model, srq_batch(valraw)), yv, NC)

    print(f"oracle real77-vs-DAS77 disc:  raw {disc0:.3f}  ->  SRQ {disc1:.3f}")
    print(f"source 7-way class probe:     raw {src0:.3f}  ->  SRQ {src1:.3f}  (drop {src0-src1:+.3f})")
    print(f"{'class':9s} {'excess raw':>11} {'excess SRQ':>11} {'drop%':>8}")
    for cc in ["Sit", "Towards", "Bend"]:
        if cc in ex0 and cc in ex1:
            dp = (ex0[cc] - ex1[cc]) / max(1e-6, ex0[cc])
            print(f"{cc:9s} {ex0[cc]:>11.3f} {ex1[cc]:>11.3f} {dp:>8.1%}")
    sit_ok = cc and (ex0["Sit"] - ex1["Sit"]) / max(1e-6, ex0["Sit"]) >= 0.25
    tow_ok = (ex0["Towards"] - ex1["Towards"]) / max(1e-6, ex0["Towards"]) >= 0.25
    print("\nGATES: disc<0.93 =", disc1 < 0.93, " Sit&Towards excess -25% =", sit_ok and tow_ok,
          " src_probe_drop<=0.03 =", (src0 - src1) <= 0.03)
    print("VERDICT SRQ:", "POTENTIAL" if (disc1 < 0.93 and sit_ok and tow_ok and (src0 - src1) <= 0.03)
          else "KILL (sensor-response canonicalization does not close the structural 77 gap)")


if __name__ == "__main__":
    main()
