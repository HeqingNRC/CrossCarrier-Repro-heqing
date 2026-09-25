#!/usr/bin/env bash
#SBATCH --job-name=xcar-anchor-exact
#SBATCH --account=st-zliu-1-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail

PKG=/scratch/st-zliu-1/heqingz/CrossCarrier_Fresh_20260922/CrossCarrier_Repro
OUT=/scratch/st-zliu-1/heqingz/CrossCarrier_Anchor_Runs_20260924
PY=/scratch/st-zliu-1/heqingz/envs/xcarrier_20260922/bin/python

cd "$PKG"

export HF_HOME="$PKG/weights"
export HUGGINGFACE_HUB_CACHE="$PKG/weights/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export PYTHONPATH="$PKG/baseline_v20:$PKG/EXPERIMENTSRESULT:$PKG"

"$PY" - <<'PY'
import json
from pathlib import Path
import importlib.util

import numpy as np
import torch

PKG = Path("/scratch/st-zliu-1/heqingz/CrossCarrier_Fresh_20260922/CrossCarrier_Repro")
OUT = Path("/scratch/st-zliu-1/heqingz/CrossCarrier_Anchor_Runs_20260924")

# ------------------------------------------------------------
# Import Zirui's ORIGINAL final-EMA aggregator.
# We reuse its exact:
#   tensor_transform
#   cache()
#   split_test77_dev_final()
#   eval_accf1()  <-- batch size 256
#   checkpoint resolution
# ------------------------------------------------------------
p = PKG / "baseline_v20/aggregate_ablation_finalema.py"

spec = importlib.util.spec_from_file_location("zirui_agg", p)
agg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agg)

# Only hardware adaptation for V100.
agg.AMP = torch.float16

DEVICE = agg.DEVICE
SEEDS = [42, 1234, 31415]

CONFIGS = [
    ("Full proposed", "A_V13_GRL"),
    ("Full DAS only", "A_REF"),
    ("Radar aug only", "E1_noDAS"),
]

print("GPU:", torch.cuda.get_device_name(0))
print("Evaluator: original aggregate_ablation_finalema.eval_accf1")
print("Eval batch size: 256")
print("AMP: fp16")

# ------------------------------------------------------------
# Exact same data construction as original aggregator
# ------------------------------------------------------------
full = agg.lib.load_manifest(
    "test",
    keep_7c=True,
).reset_index(drop=True)

y418 = full["class_idx_7c"].to_numpy()
x418 = agg.cache(full)

_dev, test278, _info = agg.lib.split_test77_dev_final(
    full,
    dev_per_class=20,
    seed=42,
)

test278 = test278.reset_index(drop=True)
y278 = test278["class_idx_7c"].to_numpy()
x278 = agg.cache(test278)

print(f"full418={len(full)} test278={len(test278)}")

rows = []
xchecks = []

for display, cfg in CONFIGS:

    base = OUT / cfg

    accs = []
    f1s = []

    print()
    print("=" * 70)
    print(cfg)
    print("=" * 70)

    for seed in SEEDS:

        seed_dir = base / f"seed{seed}"

        ckpt = agg.resolve_ckpt(seed_dir)

        if ckpt is None:
            raise RuntimeError(
                f"Missing final EMA checkpoint: {seed_dir}"
            )

        ck = torch.load(
            ckpt,
            map_location=DEVICE,
            weights_only=False,
        )

        model = agg.lib.TimmBackboneV921(
            agg.config.DEFAULT_BACKBONE,
            agg.NC,
            adapter_mode="lora",
        ).to(DEVICE).eval()

        model.load_state_dict(
            {
                k: v.to(DEVICE)
                for k, v in ck["ema"].items()
            },
            strict=False,
        )

        # EXACT original evaluator, default bs=256.
        acc418, f1418 = agg.eval_accf1(
            model,
            x418,
            y418,
        )

        acc278, f1278 = agg.eval_accf1(
            model,
            x278,
            y278,
        )

        hist_path = seed_dir / "reports/history.json"

        hist = json.load(open(hist_path))
        hist278 = float(
            hist[-1]["test77_f1_ema"]
        )

        diff = f1278 - hist278

        print(
            f"seed={seed:<5d} "
            f"418 acc={acc418:.4f} "
            f"f1={f1418:.4f} | "
            f"278 evaluator={f1278:.4f} "
            f"history={hist278:.4f} "
            f"diff={diff:+.8f}"
        )

        accs.append(acc418)
        f1s.append(f1418)

        xchecks.append({
            "config": cfg,
            "seed": seed,
            "eval278": f1278,
            "history278": hist278,
            "diff": diff,
        })

        del model
        torch.cuda.empty_cache()

    rows.append({
        "configuration": display,
        "config": cfg,
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "f1_seeds": f1s,
    })

maxdiff = max(
    abs(x["diff"])
    for x in xchecks
)

historical = {
    "A_V13_GRL": 0.832,
    "A_REF": 0.767,
    "E1_noDAS": 0.302,
}

proposed = next(
    r["f1_mean"]
    for r in rows
    if r["config"] == "A_V13_GRL"
)

for r in rows:
    r["historical_f1"] = historical[r["config"]]
    r["fresh_minus_historical_pp"] = (
        r["f1_mean"] - r["historical_f1"]
    ) * 100
    r["delta_vs_fresh_proposed_pp"] = (
        r["f1_mean"] - proposed
    ) * 100

result = {
    "evaluator": (
        "original baseline_v20/"
        "aggregate_ablation_finalema.py"
    ),
    "amp": "fp16",
    "eval_batch_size": 256,
    "rows": rows,
    "xchecks": xchecks,
    "max_abs_278_crosscheck_diff": maxdiff,
}

(OUT / "anchor_exact_full418.json").write_text(
    json.dumps(result, indent=2)
)

md = []

md.append("# Exact Fresh Anchor Evaluation")
md.append("")
md.append(
    "Original7 10+24 -> 77 GHz; "
    "original final-EMA evaluator; "
    "V100 fp16; eval batch=256."
)
md.append("")
md.append(
    "| Configuration | Accuracy | Macro-F1 | "
    "Historical F1 | Fresh-Historical | "
    "vs fresh proposed |"
)
md.append(
    "|---|---:|---:|---:|---:|---:|"
)

for r in rows:
    md.append(
        f"| {r['configuration']} | "
        f"{r['accuracy_mean']:.3f} ± "
        f"{r['accuracy_std']:.3f} | "
        f"{r['f1_mean']:.3f} ± "
        f"{r['f1_std']:.3f} | "
        f"{r['historical_f1']:.3f} | "
        f"{r['fresh_minus_historical_pp']:+.1f} pp | "
        f"{r['delta_vs_fresh_proposed_pp']:+.1f} pp |"
    )

md.append("")
md.append("## Per-seed Macro-F1")
md.append("")
md.append(
    "| Configuration | seed42 | seed1234 | seed31415 |"
)
md.append("|---|---:|---:|---:|")

for r in rows:
    v = r["f1_seeds"]
    md.append(
        f"| {r['configuration']} | "
        f"{v[0]:.4f} | "
        f"{v[1]:.4f} | "
        f"{v[2]:.4f} |"
    )

md.append("")
md.append(
    f"278-subset exact evaluator/history cross-check: "
    f"max |diff| = {maxdiff:.8f}"
)

text = "\n".join(md)

(OUT / "anchor_exact_full418.md").write_text(text)

print()
print(text)
PY
