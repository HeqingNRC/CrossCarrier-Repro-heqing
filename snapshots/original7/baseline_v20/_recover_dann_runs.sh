#!/usr/bin/env bash
# Wait for the active DANN chain to finish, then identify run_dirs whose
# training was killed mid-way (i.e., have checkpoints but no
# reports/strict_analysis.json), wipe their stale state, and re-run them.
# Finally invoke aggregate_ablation_dann.py.
set -uo pipefail

PY='/c/Users/Zirui Lin/anaconda3/envs/rader_baseline/python.exe'
ORIG_LOG=/g/zhanghe/output/V9.2.1-Ablation-DANN/_chain.log
LOG=/g/zhanghe/output/V9.2.1-Ablation-DANN/_recover.log
mkdir -p "$(dirname "$LOG")"

stage() {
    echo "[$(date '+%H:%M:%S')] === $* ===" | tee -a "$LOG"
}

stage "DANN recovery script START $(date)"

stage "Stage 0: waiting for original chain to finish"
until [ -f "$ORIG_LOG" ] && grep -q "V9.2.1 -DAS + DANN ablation chain FINISHED" "$ORIG_LOG"; do
    sleep 60
done
stage "Stage 0 DONE: original chain finished"

# Expected run-dirs and their dataset/seed mapping
declare -a TODO=(
    "self:42:/g/zhanghe/output/V9.2.1-KnownPeople-UnknownFreq/run_dann_seed42_strict"
    "self:1234:/g/zhanghe/output/V9.2.1-KnownPeople-UnknownFreq/run_dann_seed1234_strict"
    "self:7890:/g/zhanghe/output/V9.2.1-KnownPeople-UnknownFreq/run_dann_seed7890_strict"
    "ci4r:42:/g/zhanghe/output/V9.2.1-KnownPeople-UnknownFreq-CI4R/run_dann_seed42"
    "ci4r:1234:/g/zhanghe/output/V9.2.1-KnownPeople-UnknownFreq-CI4R/run_dann_seed1234"
    "ci4r:7890:/g/zhanghe/output/V9.2.1-KnownPeople-UnknownFreq-CI4R/run_dann_seed7890"
)

run_one() {
    local DATASET=$1
    local SEED=$2
    local RUN_DIR=$3
    if [ "$DATASET" = "self" ]; then
        unset V921_TASK_DIR_NAME V921_RUNS_DIR_TAG V921_EXPERIMENT_NAME
    else
        export V921_TASK_DIR_NAME=ci4r_known_people_unknown_freq
        export V921_RUNS_DIR_TAG=V9.2.1-KnownPeople-UnknownFreq-CI4R
        export V921_EXPERIMENT_NAME=V9.2.1-KnownPeople-UnknownFreq-CI4R
    fi
    export V921_USE_DAS=0
    export V921_USE_DANN=1
    export V921_DANN_WEIGHT=0.1

    stage "REDO TRAIN: ${DATASET} DANN seed=${SEED} -> ${RUN_DIR}"
    "$PY" /g/zhanghe/baseline_v9_2_1/train.py \
        --run-dir "$RUN_DIR" \
        --seed "$SEED" \
        --progress simple 2>&1 | tee -a "$LOG"

    stage "REDO ANALYZE: ${DATASET} DANN seed=${SEED}"
    "$PY" /g/zhanghe/baseline_v9_2_1/strict_analysis.py \
        --run-dir "$RUN_DIR" 2>&1 | tee -a "$LOG"
}

stage "Stage 1: scanning for failed runs (no strict_analysis.json)"
FAILED=()
for entry in "${TODO[@]}"; do
    IFS=":" read -r DATASET SEED RD <<< "$entry"
    if [ -f "${RD}/reports/strict_analysis.json" ]; then
        stage "  OK: ${DATASET} seed=${SEED}"
    else
        stage "  REDO needed: ${DATASET} seed=${SEED} -> ${RD}"
        FAILED+=("$entry")
    fi
done

if [ ${#FAILED[@]} -eq 0 ]; then
    stage "Nothing to redo. All 6 runs already complete."
else
    stage "Stage 2: wiping stale state for ${#FAILED[@]} failed runs"
    for entry in "${FAILED[@]}"; do
        IFS=":" read -r DATASET SEED RD <<< "$entry"
        if [ -d "$RD" ]; then
            rm -rf "$RD"
            stage "  wiped: $RD"
        fi
    done

    stage "Stage 3: re-training ${#FAILED[@]} failed runs"
    for entry in "${FAILED[@]}"; do
        IFS=":" read -r DATASET SEED RD <<< "$entry"
        run_one "$DATASET" "$SEED" "$RD"
    done
    unset V921_USE_DAS V921_USE_DANN V921_DANN_WEIGHT
fi

stage "Stage 4: aggregating cross-band ladder (paired Delta)"
"$PY" /g/zhanghe/baseline_v9_2_1/aggregate_ablation_dann.py 2>&1 | tee -a "$LOG"

stage "DANN recovery script FINISHED $(date)"
