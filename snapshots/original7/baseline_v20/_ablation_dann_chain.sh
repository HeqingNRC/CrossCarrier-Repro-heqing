#!/usr/bin/env bash
# Ablation: V9.2.1 with DANN cross-band signal instead of DAS curriculum.
# Runs the SAME pipeline as V9.2.1 full but:
#   USE_DAS=0  (turn off DAS curriculum)
#   USE_DANN=1 (enable band-adversarial DANN head)
# This is the "data-driven cross-band mechanism" rung of the ablation
# ladder used to argue that DAS's value comes specifically from its
# physics prior, not from any cross-band signal.
#
# 3 seeds (42, 1234, 7890) x 2 datasets (self, ci4r) = 6 runs.
set -uo pipefail

PY='/c/Users/Zirui Lin/anaconda3/envs/rader_baseline/python.exe'
LOG=/g/zhanghe/output/V9.2.1-Ablation-DANN/_chain.log
mkdir -p "$(dirname "$LOG")"

stage() {
    echo "[$(date '+%H:%M:%S')] === $* ===" | tee -a "$LOG"
}

run_dann_seed() {
    local DATASET=$1
    local SEED=$2
    local RUN_DIR
    if [ "$DATASET" = "self" ]; then
        RUN_DIR=/g/zhanghe/output/V9.2.1-KnownPeople-UnknownFreq/run_dann_seed${SEED}_strict
        unset V921_TASK_DIR_NAME V921_RUNS_DIR_TAG V921_EXPERIMENT_NAME
    else
        RUN_DIR=/g/zhanghe/output/V9.2.1-KnownPeople-UnknownFreq-CI4R/run_dann_seed${SEED}
        export V921_TASK_DIR_NAME=ci4r_known_people_unknown_freq
        export V921_RUNS_DIR_TAG=V9.2.1-KnownPeople-UnknownFreq-CI4R
        export V921_EXPERIMENT_NAME=V9.2.1-KnownPeople-UnknownFreq-CI4R
    fi
    export V921_USE_DAS=0
    export V921_USE_DANN=1
    export V921_DANN_WEIGHT=0.1

    if [ -f "${RUN_DIR}/reports/strict_analysis.json" ]; then
        stage "  SKIP: ${DATASET} DANN seed=${SEED} (already done)"
        return
    fi

    stage "TRAIN: ${DATASET} -DAS+DANN seed=${SEED} -> ${RUN_DIR}"
    "$PY" /g/zhanghe/baseline_v9_2_1/train.py \
        --run-dir "$RUN_DIR" \
        --seed "$SEED" \
        --progress simple 2>&1 | tee -a "$LOG"

    stage "ANALYZE: ${DATASET} -DAS+DANN seed=${SEED}"
    "$PY" /g/zhanghe/baseline_v9_2_1/strict_analysis.py \
        --run-dir "$RUN_DIR" 2>&1 | tee -a "$LOG"
}

stage "V9.2.1 -DAS + DANN ablation chain START $(date)"

run_dann_seed self 42
run_dann_seed self 1234
run_dann_seed self 7890
run_dann_seed ci4r 42
run_dann_seed ci4r 1234
run_dann_seed ci4r 7890

unset V921_USE_DAS V921_USE_DANN V921_DANN_WEIGHT

stage "Aggregating DANN x 3 seeds across the cross-band ladder..."
"$PY" /g/zhanghe/baseline_v9_2_1/aggregate_ablation_dann.py 2>&1 | tee -a "$LOG"
stage "V9.2.1 -DAS + DANN ablation chain FINISHED $(date)"
