#!/usr/bin/env bash
# Usage: bash hpc/submit.sh {preflight|repro|train} [hpc/config.sh]
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
MODE="${1:-}"
CONFIG_FILE="${2:-$SCRIPT_DIR/config.sh}"
case "$MODE" in
  preflight|repro|train) ;;
  *) printf 'Usage: bash hpc/submit.sh {preflight|repro|train} [config.sh]\n' >&2; exit 2 ;;
esac
[[ -f "$CONFIG_FILE" ]] || { printf 'Missing config: %s\n' "$CONFIG_FILE" >&2; exit 2; }
CONFIG_FILE="$(cd -- "$(dirname -- "$CONFIG_FILE")" && pwd -P)/$(basename -- "$CONFIG_FILE")"
# shellcheck source=/dev/null
source "$CONFIG_FILE"

for key in SOCKEYE_ACCOUNT SOCKEYE_PARTITION PROJECT_DIR OUTPUT_ROOT ENV_SETUP; do
  value="${!key:-}"
  if [[ -z "$value" || "$value" == *REPLACE_ME* ]]; then
    printf 'Set %s to the value for your own Sockeye allocation.\n' "$key" >&2
    exit 2
  fi
done
for key in PROJECT_DIR OUTPUT_ROOT ENV_SETUP; do
  [[ "${!key}" == /* ]] || { printf '%s must be an absolute Linux path.\n' "$key" >&2; exit 2; }
done
[[ -d "$PROJECT_DIR" && -f "$PROJECT_DIR/reproduce.py" ]] || {
  printf 'PROJECT_DIR does not contain reproduce.py: %s\n' "$PROJECT_DIR" >&2; exit 2;
}
[[ -r "$ENV_SETUP" ]] || { printf 'Environment activation file is missing: %s\n' "$ENV_SETUP" >&2; exit 2; }
command -v sbatch >/dev/null 2>&1 || { printf 'sbatch unavailable; run this on Sockeye.\n' >&2; exit 2; }

CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-32G}"
MAX_CONCURRENT="${MAX_CONCURRENT:-1}"
[[ "$CPUS_PER_TASK" =~ ^[1-9][0-9]*$ ]] || { printf 'CPUS_PER_TASK must be positive.\n' >&2; exit 2; }
[[ "$MAX_CONCURRENT" =~ ^[1-9]$ ]] || { printf 'MAX_CONCURRENT must be 1 through 9.\n' >&2; exit 2; }

mkdir -p -- "$OUTPUT_ROOT/slurm"
SBATCH_ARGS=(
  --parsable
  --account="$SOCKEYE_ACCOUNT"
  --partition="$SOCKEYE_PARTITION"
  --job-name="crosscarrier-$MODE"
  --nodes=1
  --ntasks=1
  --cpus-per-task="$CPUS_PER_TASK"
  --mem="$MEMORY"
  --gres=gpu:1
  --chdir="$PROJECT_DIR"
  --output="$OUTPUT_ROOT/slurm/%x-%A_%a.out"
  --error="$OUTPUT_ROOT/slurm/%x-%A_%a.err"
)
case "$MODE" in
  preflight) SBATCH_ARGS+=(--time="${PREFLIGHT_WALLTIME:-00:10:00}") ;;
  repro) SBATCH_ARGS+=(--time="${REPRO_WALLTIME:-01:00:00}") ;;
  train)
    [[ -f "$PROJECT_DIR/cross_frequency.py" ]] || { printf 'Missing cross_frequency.py.\n' >&2; exit 2; }
    case "${VARIANT:-proposed}" in
      proposed|no_das_acr) ;;
      *) printf 'VARIANT must be proposed or no_das_acr.\n' >&2; exit 2 ;;
    esac
    for target in 77 10 24; do
      for split in train val test; do
        manifest="$PROJECT_DIR/tasks/cross_frequency_grouped/target$target/manifest/$split.csv"
        [[ -f "$manifest" ]] || {
          printf 'Missing manifest: %s\nRun python prepare_cross_frequency.py --protocol grouped first.\n' "$manifest" >&2
          exit 2
        }
      done
    done
    SBATCH_ARGS+=(--time="${TRAIN_WALLTIME:-06:00:00}" --array="0-8%$MAX_CONCURRENT")
    ;;
esac

JOB_ID="$(sbatch "${SBATCH_ARGS[@]}" "$SCRIPT_DIR/gpu_job.sbatch" "$MODE" "$CONFIG_FILE")"
printf 'Submitted %s: %s\nLogs: %s/slurm\n' "$MODE" "$JOB_ID" "$OUTPUT_ROOT"
printf 'Monitor: squeue -u "$USER"\n'
printf 'Keep this config unchanged until all submitted jobs finish.\n'
