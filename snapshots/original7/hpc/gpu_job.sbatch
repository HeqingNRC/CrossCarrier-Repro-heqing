#!/usr/bin/env bash
# Resource requests are supplied by submit.sh, including exactly one GPU.
set -euo pipefail

[[ -n "${SLURM_JOB_ID:-}" ]] || {
  printf 'Run through hpc/submit.sh; do not execute GPU work on the login node.\n' >&2
  exit 2
}
MODE="${1:?Missing mode}"
CONFIG_FILE="${2:?Missing config file}"
# shellcheck source=/dev/null
source "$CONFIG_FILE"
# shellcheck source=/dev/null
source "$ENV_SETUP"
cd -- "$PROJECT_DIR"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MPLBACKEND=Agg
PYTHON_BIN="${PYTHON_BIN:-python}"
PRECISION="${PRECISION:-fp16}"
VARIANT="${VARIANT:-proposed}"

case "$MODE" in
  preflight|repro)
    RUN_DIR="$OUTPUT_ROOT/$MODE/job-$SLURM_JOB_ID"
    ;;
  train)
    TASK_ID="${SLURM_ARRAY_TASK_ID:?Training must be submitted as an array}"
    [[ "$TASK_ID" =~ ^[0-8]$ ]] || { printf 'Unexpected array task: %s\n' "$TASK_ID" >&2; exit 2; }
    TARGETS=(77 10 24)
    SEEDS=(42 1234 31415)
    TARGET="${TARGETS[$((TASK_ID / 3))]}"
    SEED="${SEEDS[$((TASK_ID % 3))]}"
    RUN_DIR="$OUTPUT_ROOT/train/job-${SLURM_ARRAY_JOB_ID}/$VARIANT/target$TARGET/seed$SEED"
    ;;
  *) printf 'Invalid mode: %s\n' "$MODE" >&2; exit 2 ;;
esac
mkdir -p -- "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/console.log") 2>&1
printf 'UTC: %s\nJob: %s\nHost: %s\nMode: %s\nProject: %s\nRun: %s\n' \
  "$(date -u +'%FT%TZ')" "$SLURM_JOB_ID" "$(hostname)" "$MODE" "$PROJECT_DIR" "$RUN_DIR"
if type module >/dev/null 2>&1; then module list 2>&1 || true; fi
"$PYTHON_BIN" -m pip freeze > "$RUN_DIR/environment.freeze.txt"
"$PYTHON_BIN" hpc/preflight.py --run-dir "$RUN_DIR"

case "$MODE" in
  preflight) printf 'GPU environment preflight passed. No model training was performed.\n' ;;
  repro)
    "$PYTHON_BIN" -u reproduce.py --n-boot "${N_BOOT:-2000}" \
      --output-root "$RUN_DIR" --batch-size "${REPRO_BATCH_SIZE:-8}" --precision "$PRECISION"
    ;;
  train)
    printf 'Target: %s GHz; seed: %s; variant: %s; precision: %s\n' "$TARGET" "$SEED" "$VARIANT" "$PRECISION"
    TRAIN_ARGS=(train --target "$TARGET" --seed "$SEED" --epochs "${EPOCHS:-100}"
      --run-dir "$RUN_DIR/model" --precision "$PRECISION" --protocol grouped --variant "$VARIANT"
      --batch-size "${TRAIN_BATCH_SIZE:-16}" --eval-batch-size "${EVAL_BATCH_SIZE:-8}"
      --num-workers "${NUM_WORKERS:-4}")
    if [[ -n "${TRAIN_MAX_STEPS:-}" ]]; then
      [[ "$TRAIN_MAX_STEPS" =~ ^[1-9][0-9]*$ ]] || { printf 'TRAIN_MAX_STEPS must be positive.\n' >&2; exit 2; }
      TRAIN_ARGS+=(--max-steps "$TRAIN_MAX_STEPS")
      printf 'SMOKE TEST: limiting steps to %s per epoch; not a final accuracy run.\n' "$TRAIN_MAX_STEPS"
    fi
    "$PYTHON_BIN" -u cross_frequency.py "${TRAIN_ARGS[@]}"
    ;;
esac
