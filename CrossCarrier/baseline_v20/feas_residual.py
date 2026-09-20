"""SPR-Morph + CR-DAS2 feasibility (NO training): does geometric DAS(10->24) leave a
STRUCTURED, class-concentrated residual vs real-24 that a renderer could learn/extrapolate?

In A_V13_GRL z_cls, linear discriminability (5-fold balanced probe; 0.5=identical):
  disc_raw  = real-10  vs real-24      (raw cross-carrier gap, no DAS)
  disc_das  = DAS(10->24) vs real-24   (residual DAS leaves on the SEEN band)
  floor     = real-24 split vs split   (sampling-noise floor)
Per-class centroid excess for DAS-vs-real24 (worst classes = where DAS misses structure).
CR-DAS2 probe: Doppler-axis Gaussian blur on DAS(10->24) at several sigmas -> does a
resolution kernel reduce disc_das (= is the residual a resolution/TF-kernel effect)?
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from PIL import Image
import config, v9_2_1lib as lib
from verify_priorities import torch_cv_probe

DEVICE = torch.device("cuda"); AMP = torch.bfloat16; NC = config.NUM_CLASSES
ROOT = config.DATASET_ROOT; REV = Path("EXPERIMENTSRESULT/REVISION_5090")
SEEDS = [42, 1234, 31415]; C = config.CLASSES
_tf = lib.tensor_transform(train=False)
np.random.seed(0)


def cache_imgs(pil_list):
    return torch.stack([_tf(im) for im in pil_list]).to(DEVICE)


def vblur(x, sigma):
    """Gaussian blur along the Doppler (height) axis only."""
    if sigma <= 0:
        return x
    k = max(3, int(6 * sigma) | 1); r = k // 2
    xs = torch.arange(k, device=x.device, dtype=torch.float32) - r
    w = torch.exp(-(xs ** 2) / (2 * sigma ** 2)); w = (w / w.sum()).view(1, 1, k, 1)
    w = w.repeat(x.shape[1], 1, 1, 1)
    return F.conv2d(x, w, padding=(r, 0), groups=x.shape[1])


@torch.no_grad()
def zcls(model, x, bs=256):
    out = []
    for i in range(0, x.shape[0], bs):
        with torch.autocast("cuda", dtype=AMP):
            out.append(model.encode_adapted(x[i:i + bs]).float().cpu())
    z = torch.cat(out).numpy()
    return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)


def main():
    tr = lib.load_manifest("train", keep_7c=True); va = lib.load_manifest("val", keep_7c=True)
    src = pd.concat([tr, va], ignore_index=True)
    pair_df, st = lib.build_exact_pairs(src)
    print(f"[pairs] {len(pair_df)} matched")
    yp = pair_df["class_idx_7c"].to_numpy()
    im10 = [Image.open(ROOT / p).convert("RGB") for p in pair_df["path_10"]]
    im24 = [Image.open(ROOT / p).convert("RGB") for p in pair_df["path_24"]]
    imdas = [lib.das_deterministic(im.copy(), 10.0, 24.0) for im in im10]
    x10 = cache_imgs(im10); x24 = cache_imgs(im24); xdas = cache_imgs(imdas)

    model = lib.TimmBackboneV921(config.DEFAULT_BACKBONE, NC, adapter_mode="lora").to(DEVICE).eval()
    agg = {"raw": [], "das": [], "floor": [], "excess": {c: [] for c in C},
           "blur": {s: [] for s in [0.5, 1.0, 2.0, 3.0]}}
    for s in SEEDS:
        ck = torch.load(REV / "A_V13_GRL" / f"seed{s}" / "checkpoints" / "pool_ep100_ema.pt",
                        map_location=DEVICE, weights_only=False)
        model.load_state_dict({k: v.to(DEVICE) for k, v in ck["ema"].items()}, strict=False)
        z10 = zcls(model, x10); z24 = zcls(model, x24); zdas = zcls(model, xdas)
        # disc real10 vs real24
        agg["raw"].append(torch_cv_probe(np.concatenate([z10, z24]),
                          np.r_[np.zeros(len(z10)), np.ones(len(z24))].astype(int), 2))
        # disc DAS(10->24) vs real24
        agg["das"].append(torch_cv_probe(np.concatenate([zdas, z24]),
                          np.r_[np.zeros(len(zdas)), np.ones(len(z24))].astype(int), 2))
        # within-real24 floor
        h = np.random.permutation(len(z24)); half = len(z24) // 2
        lab = np.zeros(len(z24), int); lab[h[:half]] = 1
        agg["floor"].append(torch_cv_probe(z24, lab, 2))
        # per-class centroid excess (das vs real24)
        for ci, cc in enumerate(C):
            a = zdas[yp == ci]; b = z24[yp == ci]
            if len(a) >= 4 and len(b) >= 4:
                ca = a.mean(0); ca /= np.linalg.norm(ca) + 1e-8
                cb = b.mean(0); cb /= np.linalg.norm(cb) + 1e-8
                agg["excess"][cc].append(float(1 - ca @ cb))
        # CR-DAS2: does a Doppler-axis blur of DAS reduce the gap to real24?
        for sig in [0.5, 1.0, 2.0, 3.0]:
            zb = zcls(model, vblur(xdas, sig))
            agg["blur"][sig].append(torch_cv_probe(np.concatenate([zb, z24]),
                                    np.r_[np.zeros(len(zb)), np.ones(len(z24))].astype(int), 2))

    m = lambda a: float(np.mean(a))
    print("\n==== structural residual in z_cls (3-seed mean discriminability, 0.5=identical) ====")
    print(f"  real10 vs real24 (raw cross-carrier) : {m(agg['raw']):.3f}")
    print(f"  DAS(10->24) vs real24 (residual)     : {m(agg['das']):.3f}   <- SPR-Morph target")
    print(f"  within-real24 floor                  : {m(agg['floor']):.3f}")
    closed = (m(agg['raw']) - m(agg['das'])) / max(1e-6, m(agg['raw']) - m(agg['floor']))
    print(f"  => DAS closes {closed:.0%} of the raw gap; residual excess over floor = {m(agg['das'])-m(agg['floor']):+.3f}")
    print("\n  per-class centroid excess (DAS vs real24), worst first:")
    for cc, ex in sorted(((cc, m(agg['excess'][cc])) for cc in C), key=lambda t: -t[1]):
        print(f"    {cc:9s} {ex:.4f}")
    print("\n==== CR-DAS2: Doppler-axis blur sigma -> DAS-vs-real24 disc (lower=blur helps) ====")
    print(f"    sigma 0 (none): {m(agg['das']):.3f}")
    for sig in [0.5, 1.0, 2.0, 3.0]:
        print(f"    sigma {sig}: {m(agg['blur'][sig]):.3f}")
    best_blur = min(m(agg['blur'][s]) for s in [0.5, 1.0, 2.0, 3.0])
    print("\nVERDICT SPR-Morph:", "POTENTIAL (structured residual exists, class-concentrated)"
          if m(agg['das']) - m(agg['floor']) > 0.15 else "KILL (DAS already matches real-24; no residual to learn)")
    print("VERDICT CR-DAS2:", f"resolution-kernel reduces disc {m(agg['das']):.3f}->{best_blur:.3f}; "
          + ("PLAUSIBLE mechanism" if best_blur < m(agg['das']) - 0.03 else "blur does NOT help -> residual is NOT a simple resolution kernel"))

    Path("EXPERIMENTSRESULT/CKPT_SELECTION_DIAG").mkdir(parents=True, exist_ok=True)
    Path("EXPERIMENTSRESULT/CKPT_SELECTION_DIAG/feas_residual.json").write_text(json.dumps(
        {"raw": m(agg['raw']), "das": m(agg['das']), "floor": m(agg['floor']),
         "excess": {cc: m(agg['excess'][cc]) for cc in C},
         "blur": {str(s): m(agg['blur'][s]) for s in [0.5, 1.0, 2.0, 3.0]}}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
