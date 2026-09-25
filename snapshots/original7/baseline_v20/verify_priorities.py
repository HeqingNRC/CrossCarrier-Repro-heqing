"""Three verifications to prioritize V13/V15 redesign (A_V20 Full, 3 seeds, final-EMA).

V1  carrier leakage in z_cls : linear probe predicts carrier(10 vs 24) from z_cls
    (kinematic neck) and from z_freq (carrier neck), on source data.
    -> z_cls probe HIGH = current split leaks carrier = GRL/adversarial split needed.
       z_cls << z_freq = split already works.

V2  sensor-head contribution : kin-only vs total(kin+sensor) vs sensor-only acc/F1
    on 77GHz-418 and source-val.
    -> total ~= kin-only AND sensor-only ~ chance = sensor head is a no-op = V15
       kin/sensor disentanglement is hollow.

V3  DAS-77 vs real-77 realism gap (z_cls feature space):
    - per-class centroid cosine distance real-77 vs DAS(source->77)
    - within-real-77 split = sampling-noise floor
    - real-vs-DAS-77 linear discriminability (balanced acc; 0.5=identical, 1=domain gap)
    -> distance >> floor / discriminability >> 0.5 = realism gap is the bottleneck.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
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
np.random.seed(0)


def cache(df, das_to=None):
    imgs = []
    for p, f in zip(df["path"], df["frequency"]):
        fs = lib.parse_freq_ghz(f)
        im = Image.open(ROOT / p).convert("RGB")
        if das_to is not None and abs(das_to - fs) > 1e-9:
            im = lib.das_deterministic(im, fs, das_to)
        imgs.append(_tf(im))
    return torch.stack(imgs).to(DEVICE)


@torch.no_grad()
def extract(model, x, bs=256):
    zc, zf, kl, sl = [], [], [], []
    for i in range(0, x.shape[0], bs):
        with torch.autocast("cuda", dtype=AMP):
            zb = model.encoder(x[i:i + bs]).float()          # skip frozen oracle
            z_cls = model.neck(zb); z_freq = model.freq_neck(zb)
            kin = model.logits_from_neck(z_cls, margin=False)
            sen = model.sensor_logits_from_neck(z_cls, margin=False)
        zc.append(z_cls.float().cpu()); zf.append(z_freq.float().cpu())
        kl.append(kin.float().cpu()); sl.append(sen.float().cpu())
    return (torch.cat(zc).numpy(), torch.cat(zf).numpy(),
            torch.cat(kl).numpy(), torch.cat(sl).numpy())


def torch_cv_probe(X, y, n_classes, k=5, steps=300):
    """Standardize + linear logistic probe, k-fold CV, return mean val accuracy."""
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(y, dtype=torch.long, device=DEVICE)
    n = len(y)
    idx = np.random.permutation(n)
    folds = np.array_split(idx, k)
    accs = []
    for f in range(k):
        va = folds[f]; tr = np.concatenate([folds[j] for j in range(k) if j != f])
        clf = torch.nn.Linear(X.shape[1], n_classes).to(DEVICE)
        opt = torch.optim.Adam(clf.parameters(), lr=1e-2, weight_decay=1e-3)
        lossf = torch.nn.CrossEntropyLoss()
        trX, trY = Xt[tr], yt[tr]
        for _ in range(steps):
            opt.zero_grad(); loss = lossf(clf(trX), trY); loss.backward(); opt.step()
        with torch.no_grad():
            pred = clf(Xt[va]).argmax(1)
            # balanced accuracy
            accs_c = []
            for c in range(n_classes):
                m = yt[va] == c
                if m.any():
                    accs_c.append(float((pred[m] == c).float().mean().cpu()))
            accs.append(float(np.mean(accs_c)))
    return float(np.mean(accs))


def accf1(logits, y):
    preds = logits.argmax(1)
    y = np.asarray(y)
    acc = float((preds == y).mean())
    f1s = []
    for c in range(NC):
        tp = int(((preds == c) & (y == c)).sum()); fp = int(((preds == c) & (y != c)).sum())
        fn = int(((preds != c) & (y == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return acc, float(np.mean(f1s))


def main():
    tr = lib.load_manifest("train", keep_7c=True).reset_index(drop=True)
    va = lib.load_manifest("val", keep_7c=True).reset_index(drop=True)
    src = __import__("pandas").concat([tr, va], ignore_index=True)
    tgt = lib.load_manifest("test", keep_7c=True).reset_index(drop=True)

    x_src = cache(src); carrier = (src["frequency"] == "24GHz").astype(int).to_numpy()
    x_val = cache(va); yv = va["class_idx_7c"].to_numpy()
    x_tgt = cache(tgt); yt = tgt["class_idx_7c"].to_numpy()
    x_das77 = cache(va, das_to=77.0)               # source-val pushed to 77 by DAS
    print(f"[data] source={len(src)} (10:{(carrier==0).sum()} 24:{(carrier==1).sum()}) "
          f"val={len(va)} target418={len(tgt)}")

    model = lib.TimmBackboneV921(config.DEFAULT_BACKBONE, NC, adapter_mode="lora").to(DEVICE).eval()

    V1 = {"zcls": [], "zfreq": []}
    V2 = {"kin418": [], "tot418": [], "sen418": [], "kinSrc": [], "senSrc": []}
    V3 = {"disc": [], "cent": {c: [] for c in C}, "floor": {c: [] for c in C}}

    for s in SEEDS:
        ck = torch.load(REV / "A_V20" / f"seed{s}" / "checkpoints" / "pool_ep100_ema.pt",
                        map_location=DEVICE, weights_only=False)
        model.load_state_dict({k: v.to(DEVICE) for k, v in ck["ema"].items()}, strict=False)

        # ---- V1: carrier leakage ----
        zc_s, zf_s, _, _ = extract(model, x_src)
        V1["zcls"].append(torch_cv_probe(zc_s, carrier, 2))
        V1["zfreq"].append(torch_cv_probe(zf_s, carrier, 2))

        # ---- V2: head contribution ----
        zc_t, zf_t, kin_t, sen_t = extract(model, x_tgt)
        V2["kin418"].append(accf1(kin_t, yt)[1])
        V2["tot418"].append(accf1(kin_t + sen_t, yt)[1])
        V2["sen418"].append(accf1(sen_t, yt)[1])
        _, _, kin_v, sen_v = extract(model, x_val)
        V2["kinSrc"].append(accf1(kin_v, yv)[1])
        V2["senSrc"].append(accf1(sen_v, yv)[1])

        # ---- V3: realism gap (z_cls space, L2-normalized to match classifier metric) ----
        zc_real = zc_t / (np.linalg.norm(zc_t, axis=1, keepdims=True) + 1e-8)
        zc_das, _, _, _ = extract(model, x_das77)
        zc_das = zc_das / (np.linalg.norm(zc_das, axis=1, keepdims=True) + 1e-8)
        # discriminability real(1) vs das77(0)
        Xd = np.concatenate([zc_real, zc_das]); yd = np.r_[np.ones(len(zc_real)), np.zeros(len(zc_das))].astype(int)
        V3["disc"].append(torch_cv_probe(Xd, yd, 2))
        # per-class centroid distance + within-real floor
        for ci, cc in enumerate(C):
            r = zc_real[yt == ci]; d = zc_das[yv == ci]
            if len(r) >= 4 and len(d) >= 1:
                cr = r.mean(0); cr /= np.linalg.norm(cr) + 1e-8
                cd = d.mean(0); cd /= np.linalg.norm(cd) + 1e-8
                V3["cent"][cc].append(float(1 - cr @ cd))
                h = np.random.permutation(len(r)); a = r[h[:len(r) // 2]]; b = r[h[len(r) // 2:]]
                ca = a.mean(0); ca /= np.linalg.norm(ca) + 1e-8
                cb = b.mean(0); cb /= np.linalg.norm(cb) + 1e-8
                V3["floor"][cc].append(float(1 - ca @ cb))

    m = lambda a: float(np.mean(a))
    print("\n================ V1: carrier leakage in z_cls (probe acc, chance~0.53) ================")
    print(f"  z_cls  -> carrier: {m(V1['zcls']):.3f}   (HIGH = z_cls still encodes carrier = split weak)")
    print(f"  z_freq -> carrier: {m(V1['zfreq']):.3f}   (should be HIGH = freq neck captures carrier)")
    print(f"  VERDICT: z_cls leakage is {'HIGH -> GRL/adversarial split warranted' if m(V1['zcls'])>0.70 else 'LOW -> split already removes carrier'}")

    print("\n================ V2: sensor-head contribution (macro-F1) ================")
    print(f"  77GHz-418: kin-only={m(V2['kin418']):.3f}  total={m(V2['tot418']):.3f}  sensor-only={m(V2['sen418']):.3f}")
    print(f"  source-val: kin-only={m(V2['kinSrc']):.3f}  sensor-only={m(V2['senSrc']):.3f}")
    gap = abs(m(V2['tot418']) - m(V2['kin418']))
    print(f"  VERDICT: total-vs-kin diff={gap:.3f}; sensor-only(src)={m(V2['senSrc']):.3f} "
          f"-> {'sensor head is ~no-op -> V15 split hollow' if (gap<0.01 and m(V2['senSrc'])<0.4) else 'sensor head carries signal'}")

    print("\n================ V3: DAS-77 vs real-77 realism gap (z_cls) ================")
    print(f"  real-vs-DAS77 discriminability (balanced acc, 0.5=identical): {m(V3['disc']):.3f}")
    print(f"  {'class':<9}{'cent_dist(real,das)':>20}{'within-real floor':>19}{'excess':>9}")
    excesses = []
    for cc in C:
        cd = m(V3['cent'][cc]); fl = m(V3['floor'][cc]); ex = cd - fl
        excesses.append((cc, ex, cd, fl))
        print(f"  {cc:<9}{cd:>20.4f}{fl:>19.4f}{ex:>9.4f}")
    excesses.sort(key=lambda t: -t[1])
    print(f"  VERDICT: discriminability {m(V3['disc']):.3f} "
          f"({'HIGH -> DAS-77 != real-77 = realism gap real' if m(V3['disc'])>0.7 else 'LOW -> DAS-77 ~ real-77'}); "
          f"worst classes: {[e[0] for e in excesses[:3]]}")

    out = {"V1": {k: m(v) for k, v in V1.items()},
           "V2": {k: m(v) for k, v in V2.items()},
           "V3_disc": m(V3["disc"]),
           "V3_excess": {cc: m(V3['cent'][cc]) - m(V3['floor'][cc]) for cc in C}}
    Path("EXPERIMENTSRESULT/CKPT_SELECTION_DIAG/verify_priorities.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print("\n[done] wrote verify_priorities.json")


if __name__ == "__main__":
    main()
