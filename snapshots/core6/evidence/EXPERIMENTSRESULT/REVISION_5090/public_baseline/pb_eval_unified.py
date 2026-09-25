"""UNIFIED evaluator for the public-baseline table.

For EVERY method it loads the final-EMA model and forwards on the SAME two sets
(full-418 target 77 GHz, and source-val 10/24 GHz) through ONE identical metric
path (pb_lib.eval_forward == aggregate_ablation_finalema.eval_accf1). So every
number in the table comes from the same harness; differences are method only.

Model families handled:
  generic   -- GenericBackbone(key); ck["ema"] (strict);            forward = model(x_std)
  proposed  -- TimmBackboneV921(lora); ck["ema"] (strict=False);    forward = logits_from_neck(encode_adapted(x_std), margin=False)
  external  -- (#7 RadMamba / #8 SelaFD) plugged in when/if they run; else reported un-run.

Rows (single-model, head-to-head #1-10) + a SEPARATE ensemble row (#11):
  1 vgg16_bn  2 mobilenetv3_l  3 effnetb0  4 convnext_t  5 swin_t  6 convnextv2_t   [generic]
  7 RadMamba  8 SelaFD                                                              [external]
  9 E1_noDAS (DINOv3-LoRA, DAS/ACR OFF)                                             [proposed family, reuse]
 10 A_V13_GRL (Proposed single = DAS + ACR)                                         [proposed family, reuse]
 11 A_V13_GRL 3-seed posterior ensemble (deployment) -- bootstrap CI, NOT head-to-head

Cross-check: for the proposed family we also score the 278-subset final-EMA and
compare to each run's history.json last-epoch test77_f1_ema (must stay ~0.0000).

Outputs (machine + human):
  public_baseline/pb_results.json
  public_baseline/pb_results_auto.md   (drop-in table to fold into PUBLIC_BASELINE_RESULTS.md)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import pb_lib as P

REV = P.REPO / "EXPERIMENTSRESULT" / "REVISION_5090"
PB = REV / "public_baseline"
RUNS = PB / "runs"

def make_registry(runs_root):
    # (#, key, label, family, dir-template relative to REV, head_to_head)
    return [
    (1, "vgg16_bn",      "VGG16-BN",                 "generic",  runs_root / "vgg16_bn",      True),
    (2, "mobilenetv3_l", "MobileNetV3-Large",        "generic",  runs_root / "mobilenetv3_l", True),
    (3, "effnetb0",      "EfficientNet-B0",          "generic",  runs_root / "effnetb0",      True),
    (4, "convnext_t",    "ConvNeXt-Tiny",            "generic",  runs_root / "convnext_t",    True),
    (5, "swin_t",        "Swin-Tiny",                "generic",  runs_root / "swin_t",        True),
    (6, "convnextv2_t",  "ConvNeXtV2-Tiny",          "generic",  runs_root / "convnextv2_t",  True),
    (7, "radmamba",      "RadMamba",                 "external", runs_root / "radmamba",      True),
    (8, "selafd",        "SelaFD-ViT-B/16",          "external", runs_root / "selafd",        True),
    (9, "E1_noDAS",      "DINOv3 ViT-L/16", "proposed", REV / "E1_noDAS", True),
    (10, "A_V13_GRL",    "proposed", "proposed", REV / "A_V13_GRL", True),
    ]
ENSEMBLE = (11, "A_V13_GRL", "Proposed + 3-seed gap-aware ensemble (deployment)", REV / "A_V13_GRL")

CITED = {  # do-not-recompute anchors from ABLATION_REPORT.md (printed for sanity next to recompute)
    10: {"macro_f1": "0.832 +/- 0.034", "acc": "0.836 +/- 0.032", "per_seed": "0.791/0.832/0.874"},
    11: {"macro_f1": "0.857", "acc": "0.859"},
}


# --------------------------- forward builders ---------------------------
def generic_forward(backbone_key, ckpt_path):
    ck = torch.load(ckpt_path, map_location=P.DEVICE, weights_only=False)
    model_cfg = ck.get("model_cfg", {})
    model = P.GenericBackbone(backbone_key, pretrained=False, **model_cfg).to(P.DEVICE).eval()
    model.load_state_dict({k: v.to(P.DEVICE) for k, v in ck["ema"].items()}, strict=True)
    n_train = int(ck.get("n_trainable", sum(p.numel() for p in model.parameters())))
    return (lambda x: model(x)), n_train, model


_PROPOSED_MODEL = None
def proposed_forward(ckpt_path):
    global _PROPOSED_MODEL
    if _PROPOSED_MODEL is None:
        _PROPOSED_MODEL = P.lib.TimmBackboneV921(
            P.config.DEFAULT_BACKBONE, P.NC, adapter_mode="lora").to(P.DEVICE).eval()
    model = _PROPOSED_MODEL
    ck = torch.load(ckpt_path, map_location=P.DEVICE, weights_only=False)
    model.load_state_dict({k: v.to(P.DEVICE) for k, v in ck["ema"].items()}, strict=False)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fwd = lambda x: model.logits_from_neck(model.encode_adapted(x), margin=False)
    return fwd, n_train, model


def external_forward(key, ckpt_path):
    import pb_external
    ck = torch.load(ckpt_path, map_location=P.DEVICE, weights_only=False)
    model_cfg = ck.get("model_cfg", {})
    model = pb_external.build_model(key, P.NC, model_cfg=model_cfg).to(P.DEVICE).eval()
    model.load_state_dict({k: v.to(P.DEVICE) for k, v in ck["ema"].items()}, strict=True)
    n_train = int(ck.get("n_trainable", sum(p.numel() for p in model.parameters() if p.requires_grad)))
    return (lambda x: model(x)), n_train, model


def build_forward(key, family, ckpt_path):
    if family == "generic":
        return generic_forward(key, ckpt_path)
    if family == "proposed":
        return proposed_forward(ckpt_path)
    if family == "external":
        return external_forward(key, ckpt_path)
    raise ValueError(family)


def softmax_posteriors(forward_fn, batches):
    out = []
    with torch.no_grad():
        for x, _y, _d in batches:
            with torch.autocast("cuda", dtype=P.AMP):
                lg = forward_fn(x)
            out.append(torch.softmax(lg.float(), dim=1).cpu())
    return torch.cat(out).numpy()


# --------------------------- main ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--latency", action="store_true", help="also benchmark batch=1 bf16 latency")
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these method keys")
    ap.add_argument("--runs-root", default=str(RUNS),
                    help="native public-baseline runs root; use public_baseline/runs_strong for optimized baselines.")
    ap.add_argument("--out-prefix", default="pb_results",
                    help="output prefix under public_baseline, e.g. pb_results_strong")
    args = ap.parse_args()
    runs_root = Path(args.runs_root)
    registry = make_registry(runs_root)

    train_df, val_df, test_df = P.load_manifests()
    test_batches = P.build_eval_batches(test_df)            # full 418
    val_batches = P.build_eval_batches(val_df)              # source-val
    y418 = np.concatenate([y.cpu().numpy() for _x, y, _d in test_batches])
    # 278-subset (for the proposed-family harness cross-check vs history.json)
    _dev, test278, _info = P.lib.split_test77_dev_final(test_df, dev_per_class=20, seed=42)
    test278_batches = P.build_eval_batches(test278.reset_index(drop=True))
    print(f"[data] full-418={len(test_df)} source-val={len(val_df)} 278-subset={len(test278)}")

    results = []
    xcheck = []
    for num, key, label, family, base, h2h in registry:
        if args.only and key not in args.only:
            continue
        per_seed = {"acc": [], "f1": [], "src_f1": [], "src_acc": [], "gap": []}
        per_class_f1_acc = []
        n_train = None
        ran = True
        first_preds = None
        for s in P.SEEDS:
            ck = base / f"seed{s}" / "checkpoints" / "pool_ep100_ema.pt"
            if not ck.exists():
                ran = False
                break
            try:
                fwd, n_train, _m = build_forward(key, family, ck)
            except Exception as e:
                print(f"#{num} {key} seed{s}: build/load failed -> {type(e).__name__}: {e}")
                ran = False
                break
            tgt = P.eval_forward(fwd, test_batches)
            src = P.eval_forward(fwd, val_batches)
            per_seed["acc"].append(tgt["acc"]); per_seed["f1"].append(tgt["macro_f1"])
            per_seed["src_acc"].append(src["acc"]); per_seed["src_f1"].append(src["macro_f1"])
            per_seed["gap"].append(src["macro_f1"] - tgt["macro_f1"])
            per_class_f1_acc.append(tgt["per_class_f1"])
            if first_preds is None:
                first_preds = tgt["preds"]
            if family == "proposed":
                t278 = P.eval_forward(fwd, test278_batches)
                hp = base / f"seed{s}" / "reports" / "history.json"
                if hp.exists():
                    hist = json.loads(hp.read_text(encoding="utf-8"))
                    last = [r for r in hist if "test77_f1_ema" in r]
                    if last:
                        hv = float(last[-1]["test77_f1_ema"])
                        xcheck.append((key, s, t278["macro_f1"], hv, t278["macro_f1"] - hv))

        if not ran:
            results.append({"num": num, "key": key, "label": label, "family": family,
                            "head_to_head": h2h, "status": "NOT RUN",
                            "cited": CITED.get(num)})
            print(f"#{num:2d} {label:45s}  NOT RUN (missing checkpoints under {base})")
            continue

        def ms(a):
            return float(np.mean(a)), float(np.std(a))
        f1m, f1s = ms(per_seed["f1"]); accm, accs = ms(per_seed["acc"])
        sf1m, sf1s = ms(per_seed["src_f1"]); gm, gs = ms(per_seed["gap"])
        pcf1 = np.mean(np.array(per_class_f1_acc), axis=0).tolist()
        ci_lo, ci_hi, _ = P.bootstrap_ci_macrof1(first_preds, y418, n_boot=args.n_boot)
        lat = P.latency_ms_per_sample(
            build_forward(key, family, base / "seed42" / "checkpoints" / "pool_ep100_ema.pt")[0]
        ) if args.latency else None
        row = {
            "num": num, "key": key, "label": label, "family": family, "head_to_head": h2h,
            "status": "ok",
            "macro_f1_mean": f1m, "macro_f1_std": f1s,
            "acc_mean": accm, "acc_std": accs,
            "src_f1_mean": sf1m, "src_f1_std": sf1s,
            "gap_mean": gm, "gap_std": gs,
            "f1_seeds": [round(v, 4) for v in per_seed["f1"]],
            "acc_seeds": [round(v, 4) for v in per_seed["acc"]],
            "per_class_f1": [round(v, 4) for v in pcf1],
            "worst_class_f1": float(min(pcf1)),
            "boot95_lo": ci_lo, "boot95_hi": ci_hi,
            "n_trainable": int(n_train), "n_trainable_M": round(n_train / 1e6, 3),
            "latency_ms_b1_bf16": (round(lat[0], 2) if lat else None),
            "cited": CITED.get(num),
        }
        results.append(row)
        print(f"#{num:2d} {label:45s} f1={f1m:.4f}+/-{f1s:.4f} acc={accm:.4f} "
              f"src_f1={sf1m:.4f} gap={gm:+.4f} ci95=[{ci_lo:.3f},{ci_hi:.3f}] "
              f"params={n_train/1e6:.2f}M seeds={row['f1_seeds']}")

    # ---- ensemble row (#11): average the 3 proposed-seed posteriors on 418 ----
    ens = None
    enum, ekey, elabel, ebase = ENSEMBLE
    if not (args.only and ekey not in args.only):
        cks = [ebase / f"seed{s}" / "checkpoints" / "pool_ep100_ema.pt" for s in P.SEEDS]
        if all(c.exists() for c in cks):
            post = np.zeros((len(y418), P.NC))
            for c in cks:
                fwd, _n, _m = proposed_forward(c)
                post += softmax_posteriors(fwd, test_batches)
            post /= len(cks)
            preds = post.argmax(1)
            acc, mf1, pcf1, _p, _r = P.acc_macrof1_perclass(preds, y418)
            ci_lo, ci_hi, _ = P.bootstrap_ci_macrof1(preds, y418, n_boot=args.n_boot)
            ens = {"num": enum, "key": ekey, "label": elabel, "head_to_head": False,
                   "status": "ok", "macro_f1": mf1, "acc": acc,
                   "per_class_f1": [round(v, 4) for v in pcf1],
                   "boot95_lo": ci_lo, "boot95_hi": ci_hi, "cited": CITED.get(enum)}
            print(f"#{enum:2d} {elabel:45s} f1={mf1:.4f} acc={acc:.4f} "
                  f"ci95=[{ci_lo:.3f},{ci_hi:.3f}] (cited 0.857/0.859)")
        else:
            ens = {"num": enum, "key": ekey, "label": elabel, "status": "NOT RUN",
                   "cited": CITED.get(enum)}

    maxd = max((abs(d) for *_x, d in xcheck), default=0.0)
    out = {"results": results, "ensemble": ens,
           "xcheck_max_abs_diff": maxd, "xcheck": xcheck,
           "classes": list(P.config.CLASSES), "n_target": int(len(y418)),
           "n_source_val": int(len(val_df))}
    out_json = PB / f"{args.out_prefix}.json"
    out_md = PB / f"{args.out_prefix}_auto.md"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    write_md(out, out_md)
    print(f"\n[xcheck] proposed-family 278 vs history max|diff| = {maxd:.4f} (want ~0.0000)")
    print(f"saved {out_json} and {out_md}")


def write_md(out, out_path):
    cls = out["classes"]
    L = []
    L.append("# Public baseline -- auto-generated results (fold into PUBLIC_BASELINE_RESULTS.md)\n")
    L.append(f"Target = full {out['n_target']}-image 77 GHz; source-val n={out['n_source_val']}. "
             "Final-EMA (ep100), 3 seeds (42/1234/31415). One unified metric path.\n")
    L.append("## Main table -- SINGLE-model, head-to-head (#1-10)\n")
    L.append("| # | Method | 77GHz macro-F1 | 77GHz acc | src-val F1 | gen. gap | params (M) | latency ms | boot95 F1 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in out["results"]:
        if r["status"] != "ok":
            cite = f" (cited {r['cited']['macro_f1']})" if r.get("cited") else ""
            L.append(f"| {r['num']} | {r['label']} | **NOT RUN**{cite} | - | - | - | - | - | - |")
            continue
        lat = f"{r['latency_ms_b1_bf16']}" if r["latency_ms_b1_bf16"] else "-"
        L.append(f"| {r['num']} | {r['label']} | {r['macro_f1_mean']:.3f} ± {r['macro_f1_std']:.3f} "
                 f"| {r['acc_mean']:.3f} ± {r['acc_std']:.3f} | {r['src_f1_mean']:.3f} ± {r['src_f1_std']:.3f} "
                 f"| {r['gap_mean']:+.3f} | {r['n_trainable_M']:.2f} | {lat} "
                 f"| [{r['boot95_lo']:.3f}, {r['boot95_hi']:.3f}] |")
    if out.get("ensemble"):
        e = out["ensemble"]
        L.append("\n## Deployment row -- SEPARATE, NOT head-to-head (#11)\n")
        if e["status"] == "ok":
            L.append("| # | Method | 77GHz macro-F1 | 77GHz acc | boot95 F1 |")
            L.append("|---|---|---|---|---|")
            L.append(f"| {e['num']} | {e['label']} | {e['macro_f1']:.3f} | {e['acc']:.3f} "
                     f"| [{e['boot95_lo']:.3f}, {e['boot95_hi']:.3f}] |  (cited 0.857/0.859)")
        else:
            L.append("Ensemble NOT RUN (missing proposed checkpoints).")
    L.append("\n## Per-class macro-F1 (3-seed mean, full-418)\n")
    L.append("| Method | " + " | ".join(cls) + " | worst |")
    L.append("|---|" + "|".join(["---"] * (len(cls) + 1)) + "|")
    for r in out["results"]:
        if r["status"] != "ok":
            continue
        L.append(f"| {r['label']} | " + " | ".join(f"{v:.2f}" for v in r["per_class_f1"])
                 + f" | {r['worst_class_f1']:.2f} |")
    L.append(f"\n## Harness cross-check\nproposed-family 278-subset vs history.json "
             f"max|diff| = **{out['xcheck_max_abs_diff']:.4f}** (want ~0.0000)\n")
    out_path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
