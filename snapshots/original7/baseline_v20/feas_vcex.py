"""VCE-X kill-test: does the 3-seed logit ensemble's gain concentrate on Sit/Kneel
and the top-3 sink confusions (mechanism), or is it a generic average (baseline only)?

PASS iff (a) ensemble macro-F1 >= 0.857-0.005 AND acc >= 0.859-0.005, AND
         (b) top-3 sink confusion RATE drops >=10% vs pooled single-seed, AND
             (Sit F1 +>=0.05 vs 0.722  OR  Sit recall +>=0.05 vs 0.637 with prec>=0.807).
"""
from __future__ import annotations
import numpy as np, torch
from pathlib import Path
from PIL import Image
import config, v9_2_1lib as lib

DEVICE = torch.device("cuda"); AMP = torch.bfloat16; NC = config.NUM_CLASSES
ROOT = config.DATASET_ROOT; REV = Path("EXPERIMENTSRESULT/REVISION_5090")
SEEDS = [42, 1234, 31415]; C = config.CLASSES
IDX = {c: i for i, c in enumerate(C)}
_tf = lib.tensor_transform(train=False)
SINKS = [("Bend", "Pick"), ("Sit", "Kneel"), ("Towards", "SStep")]


def cache(df):
    return torch.stack([_tf(Image.open(ROOT / p).convert("RGB")) for p in df["path"]]).to(DEVICE)


@torch.no_grad()
def logits(model, x, bs=256):
    out = []
    for i in range(0, x.shape[0], bs):
        with torch.autocast("cuda", dtype=AMP):
            out.append(model.logits_from_neck(model.encode_adapted(x[i:i + bs]), margin=False).float().cpu())
    return torch.cat(out).numpy()


def prf(pred, y):
    P, R, Fn = {}, {}, {}
    for c in range(NC):
        tp = int(((pred == c) & (y == c)).sum()); fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
        P[C[c]] = p; R[C[c]] = r; Fn[C[c]] = 2 * p * r / (p + r) if p + r else 0.0
    return P, R, Fn


def conf(pred, y):
    m = np.zeros((NC, NC), int)
    for t, p in zip(y, pred):
        m[t, p] += 1
    return m


def main():
    full = lib.load_manifest("test", keep_7c=True).reset_index(drop=True)
    y = full["class_idx_7c"].to_numpy(); x = cache(full)
    model = lib.TimmBackboneV921(config.DEFAULT_BACKBONE, NC, adapter_mode="lora").to(DEVICE).eval()
    L = []
    pooled_conf = np.zeros((NC, NC), int); seed_f1 = []; seed_sit = {"f1": [], "rec": [], "prec": []}
    for s in SEEDS:
        ck = torch.load(REV / "A_V13_GRL" / f"seed{s}" / "checkpoints" / "pool_ep100_ema.pt",
                        map_location=DEVICE, weights_only=False)
        model.load_state_dict({k: v.to(DEVICE) for k, v in ck["ema"].items()}, strict=False)
        lg = logits(model, x); L.append(lg); pr = lg.argmax(1)
        P, R, Fn = prf(pr, y); pooled_conf += conf(pr, y)
        seed_f1.append(float(np.mean(list(Fn.values()))))
        seed_sit["f1"].append(Fn["Sit"]); seed_sit["rec"].append(R["Sit"]); seed_sit["prec"].append(P["Sit"])

    ens = np.mean(L, axis=0); epr = ens.argmax(1)
    eP, eR, eF = prf(epr, y); econf = conf(epr, y)
    ens_f1 = float(np.mean(list(eF.values()))); ens_acc = float((epr == y).mean())

    print(f"single-seed macro-F1 = {np.mean(seed_f1):.4f}  | ENSEMBLE macro-F1 = {ens_f1:.4f}  acc = {ens_acc:.4f}")
    print("\nper-class F1: single-seed mean -> ensemble (delta)")
    for c in C:
        sm = np.mean([prf(L[i].argmax(1), y)[2][c] for i in range(3)])
        print(f"  {c:9s} {sm:.3f} -> {eF[c]:.3f}  ({eF[c]-sm:+.3f})")

    print("\ntop-3 sink confusions (rate = count / N):")
    Npool = 3 * len(y); Nens = len(y)
    pooled_top3 = sum(pooled_conf[IDX[a], IDX[b]] for a, b in SINKS)
    ens_top3 = sum(econf[IDX[a], IDX[b]] for a, b in SINKS)
    for a, b in SINKS:
        print(f"  {a}->{b}: pooled {pooled_conf[IDX[a],IDX[b]]}/{Npool}={pooled_conf[IDX[a],IDX[b]]/Npool:.3f}"
              f"   ens {econf[IDX[a],IDX[b]]}/{Nens}={econf[IDX[a],IDX[b]]/Nens:.3f}")
    pooled_rate = pooled_top3 / Npool; ens_rate = ens_top3 / Nens
    drop = (pooled_rate - ens_rate) / pooled_rate
    print(f"  TOTAL top-3 sink rate: pooled {pooled_rate:.3f} -> ens {ens_rate:.3f}  (drop {drop:+.1%})")

    sit_f1_sm = np.mean(seed_sit["f1"]); sit_rec_sm = np.mean(seed_sit["rec"])
    print(f"\nSit: single-seed F1 {sit_f1_sm:.3f} -> ens {eF['Sit']:.3f} ({eF['Sit']-sit_f1_sm:+.3f}); "
          f"recall {sit_rec_sm:.3f} -> {eR['Sit']:.3f} ({eR['Sit']-sit_rec_sm:+.3f}); ens prec {eP['Sit']:.3f}")

    perf_ok = ens_f1 >= 0.857 - 0.005 and ens_acc >= 0.859 - 0.005
    mech_sink = drop >= 0.10
    mech_sit = (eF["Sit"] - sit_f1_sm >= 0.05) or (eR["Sit"] - sit_rec_sm >= 0.05 and eP["Sit"] >= 0.807)
    print(f"\nGATES: perf_reproduces={perf_ok}  sink_rate_drop>=10%={mech_sink}  Sit_mechanism={mech_sit}")
    print("VERDICT VCE-X:",
          "MECHANISM CONFIRMED (keep as variance-marginalization contribution)" if (perf_ok and mech_sink and mech_sit)
          else "PERF reproduces but MECHANISM NOT clean -> strong baseline, NOT a science claim" if perf_ok
          else "FAIL to reproduce -> harness/impl issue")


if __name__ == "__main__":
    main()
