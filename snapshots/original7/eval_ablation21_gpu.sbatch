#!/usr/bin/env bash
#SBATCH --job-name=xcar-ab21-eval
#SBATCH --account=st-zliu-1-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00

set -euo pipefail

PKG=/scratch/st-zliu-1/heqingz/CrossCarrier_Fresh_20260922/CrossCarrier_Repro
PY=/scratch/st-zliu-1/heqingz/envs/xcarrier_20260922/bin/python

cd "$PKG"

export HF_HOME="$PKG/weights"
export HUGGINGFACE_HUB_CACHE="$PKG/weights/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export PYTHONPATH="$PKG/baseline_v20:$PKG/EXPERIMENTSRESULT:$PKG"
export AGG_INCLUDE_SUPP=1

mkdir -p EXPERIMENTSRESULT/CKPT_SELECTION_DIAG

"$PY" - <<'PY'
import os
import torch
import importlib.util
from pathlib import Path

p = Path("baseline_v20/aggregate_ablation_finalema.py")

spec = importlib.util.spec_from_file_location("agg", p)
agg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agg)

# V100
agg.AMP = torch.float16

# Evaluate ONLY the 7 newly trained supplementary ablations.
# Do not waste time evaluating missing historical A_V20/A_REF/etc.
agg.CONFIGS = []

os.environ["AGG_INCLUDE_SUPP"] = "1"

print("GPU:", torch.cuda.get_device_name(0))
print("Evaluating 7 configs x 3 seeds on full-418...")
agg.main()
PY
