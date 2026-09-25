#!/usr/bin/env bash
set -Eeuo pipefail
KIT="${1:?toolkit path required}"
MODE="${2:?stage required}"
BATCH="${3:?batch path required}"
PY="${4:?python path required}"
if type module >/dev/null 2>&1; then
    module load miniconda3
fi
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_TELEMETRY=1
CACHE="${SLURM_TMPDIR:-/tmp}/core6-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
export MPLCONFIGDIR="$CACHE/mpl" XDG_CACHE_HOME="$CACHE/xdg"
export PYTORCH_KERNEL_CACHE_PATH="$CACHE/kernels" PIP_CACHE_DIR="$CACHE/pip"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$PYTORCH_KERNEL_CACHE_PATH" "$PIP_CACHE_DIR"
cd "$KIT"
echo "UTC=$(date -u +%FT%TZ) HOST=$(hostname) JOB=$SLURM_JOB_ID ARRAY=$SLURM_ARRAY_JOB_ID TASK=$SLURM_ARRAY_TASK_ID MODE=$MODE"
exec "$PY" "$KIT/suite.py" "$MODE" --batch "$BATCH"
