#!/usr/bin/env bash
set -euo pipefail

PKG=/scratch/st-zliu-1/heqingz/CrossCarrier_Fresh_20260922/CrossCarrier_Repro
PY=/scratch/st-zliu-1/heqingz/envs/xcarrier_20260922/bin/python
OUT=/scratch/st-zliu-1/heqingz/CrossCarrier_Anchor_Runs_20260924

cd "$PKG"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

export HF_HOME="$PKG/weights"
export HUGGINGFACE_HUB_CACHE="$PKG/weights/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export PYTHONPATH="$PKG/baseline_v20:$PKG/EXPERIMENTSRESULT:$PKG"

export V921_TASK_DIR_NAME=known_people_unknown_freq
export V921_TASK_ROOT="$PKG/tasks/known_people_unknown_freq"
export V921_DATASET_ROOT="$PKG"

export V921_TRAIN_FREQS="10GHz,24GHz"
export V921_TEST_FREQS="77GHz"
export V921_AMP_DTYPE=fp16

"$PY" - <<'PY'
import json
import numpy as np
import torch
from pathlib import Path
import sys

PKG = Path("/scratch/st-zliu-1/heqingz/CrossCarrier_Fresh_20260922/CrossCarrier_Repro")
OUT = Path("/scratch/st-zliu-1/heqingz/CrossCarrier_Anchor_Runs_20260924")

sys.path.insert(0, str(PKG / "baseline_v20"))
sys.path.insert(0, str(PKG / "EXPERIMENTSRESULT"))
sys.path.insert(0, str(PKG))

import config
import v9_2_1lib as lib

DEVICE = torch.device("cuda")
AMP = torch.float16
SEEDS = [42, 1234, 31415]

CONFIGS = [
    ("Full proposed", "A_V13_GRL"),
    ("Full DAS only", "A_REF"),
    ("Radar aug only", "E1_noDAS"),
]

print("GPU:", torch.cuda.get_device_name(0), flush=True)

# ------------------------------------------------------------
# Full 418 target
# ------------------------------------------------------------

df418 = lib.load_manifest("test", keep_7c=True).reset_index(drop=True)
y418 = df418["class_idx_7c"].to_numpy()

tf = lib.tensor_transform(train=False)

from PIL import Image

x418 = torch.stack([
    tf(Image.open(config.DATASET_ROOT / p).convert("RGB"))
    for p in df418["path"]
]).to(DEVICE)

# Historical internal 278 subset, for evaluator cross-check
_dev, df278, _info = lib.split_test77_dev_final(
    df418,
    dev_per_class=20,
    seed=42,
)
df278 = df278.reset_index(drop=True)
y278 = df278["class_idx_7c"].to_numpy()

x278 = torch.stack([
    tf(Image.open(config.DATASET_ROOT / p).convert("RGB"))
    for p in df278["path"]
]).to(DEVICE)


@torch.no_grad()
def evaluate(model, x, y, bs=128):
    preds = []

    for i in range(0, len(y), bs):
        with torch.autocast("cuda", dtype=AMP):
            z = model.encode_adapted(x[i:i+bs])
            logits = model.logits_from_neck(
                z,
                labels=None,
                margin=False,
            )

        preds.append(
            logits.float().argmax(1).cpu()
        )

    pred = torch.cat(preds).numpy()
    y = np.asarray(y)

    acc = float((pred == y).mean())

    f1s = []

    for c in range(config.NUM_CLASSES):
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())

        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0

        f1 = (
            2 * p * r / (p + r)
            if p + r
            else 0.0
        )

        f1s.append(f1)

    return acc, float(np.mean(f1s))


model = lib.TimmBackboneV921(
    config.DEFAULT_BACKBONE,
    config.NUM_CLASSES,
    adapter_mode="lora",
).to(DEVICE).eval()

rows = []
xchecks = []

for display, cfg in CONFIGS:

    f418s = []
    a418s = []

    print()
    print("==========", cfg, "==========")

    for seed in SEEDS:

        run = OUT / cfg / f"seed{seed}"
        ckpt = run / "epoch_ckpts" / "ep100.pt"

        if not ckpt.is_file():
            raise RuntimeError(f"Missing ep100 checkpoint: {ckpt}")

        ck = torch.load(
            ckpt,
            map_location=DEVICE,
            weights_only=False,
        )

        model.load_state_dict(
            {
                k: v.to(DEVICE)
                for k, v in ck["ema"].items()
            },
            strict=False,
        )

        acc418, f1418 = evaluate(
            model,
            x418,
            y418,
        )

        acc278, f1278 = evaluate(
            model,
            x278,
            y278,
        )

        history = json.load(
            open(run / "reports" / "history.json")
        )

        hist278 = float(
            history[-1]["test77_f1_ema"]
        )

        diff = f1278 - hist278

        print(
            f"seed{seed}: "
            f"full418 acc/f1={acc418:.4f}/{f1418:.4f} "
            f"278={f1278:.4f} "
            f"history={hist278:.4f} "
            f"diff={diff:+.6f}"
        )

        a418s.append(acc418)
        f418s.append(f1418)

        xchecks.append({
            "config": cfg,
            "seed": seed,
            "eval278": f1278,
            "history278": hist278,
            "diff": diff,
        })

    rows.append({
        "variant": display,
        "config": cfg,
        "acc_mean": float(np.mean(a418s)),
        "acc_std": float(np.std(a418s)),
        "f1_mean": float(np.mean(f418s)),
        "f1_std": float(np.std(f418s)),
        "f1_seeds": f418s,
        "acc_seeds": a418s,
    })


proposed = next(
    r["f1_mean"]
    for r in rows
    if r["config"] == "A_V13_GRL"
)

for r in rows:
    r["delta_vs_proposed_pp"] = (
        r["f1_mean"] - proposed
    ) * 100


# Historical reference from Zirui's report
historical = {
    "A_V13_GRL": 0.832,
    "A_REF": 0.767,
    "E1_noDAS": 0.302,
}

for r in rows:
    r["historical_f1"] = historical[r["config"]]
    r["fresh_minus_historical_pp"] = (
        r["f1_mean"] - historical[r["config"]]
    ) * 100


maxdiff = max(
    abs(x["diff"])
    for x in xchecks
)

result = {
    "protocol": "original7 10+24 -> 77, final ep100 EMA",
    "device": torch.cuda.get_device_name(0),
    "precision": "fp16",
    "rows": rows,
    "xcheck_max_abs_diff": maxdiff,
    "xchecks": xchecks,
}

(OUT / "anchor_full418.json").write_text(
    json.dumps(result, indent=2)
)


md = []

md.append(
    "# Fresh Anchor Ablation — Original7 full-418"
)
md.append("")
md.append(
    "10+24 GHz -> 77 GHz; seeds 42/1234/31415; "
    "100 epochs; final epoch EMA; V100 fp16."
)
md.append("")
md.append(
    "| Configuration | Accuracy | Macro-F1 | "
    "Historical F1 | Fresh-Historical | "
    "Delta vs fresh proposed |"
)
md.append(
    "|---|---:|---:|---:|---:|---:|"
)

for r in rows:
    md.append(
        f"| {r['variant']} | "
        f"{r['acc_mean']:.3f} ± {r['acc_std']:.3f} | "
        f"{r['f1_mean']:.3f} ± {r['f1_std']:.3f} | "
        f"{r['historical_f1']:.3f} | "
        f"{r['fresh_minus_historical_pp']:+.1f} pp | "
        f"{r['delta_vs_proposed_pp']:+.1f} pp |"
    )

md.append("")
md.append("## Per-seed Macro-F1")
md.append("")
md.append(
    "| Configuration | seed42 | seed1234 | seed31415 |"
)
md.append("|---|---:|---:|---:|")

for r in rows:
    vals = r["f1_seeds"]
    md.append(
        f"| {r['variant']} | "
        f"{vals[0]:.4f} | "
        f"{vals[1]:.4f} | "
        f"{vals[2]:.4f} |"
    )

md.append("")
md.append(
    f"278-subset evaluator/history cross-check: "
    f"max |diff| = {maxdiff:.6f}"
)

text = "\n".join(md)

(OUT / "anchor_full418.md").write_text(text)

print()
print(text)
PY

echo
echo "ANCHOR_EVAL_COMPLETE"
