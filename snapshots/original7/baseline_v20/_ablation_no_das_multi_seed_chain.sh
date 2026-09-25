#!/usr/bin/env bash
# Run V9.2.1 -DAS for seeds 1234 and 7890 on both datasets so that the
# -DAS column matches the full-V9.2.1 column's 3-seed setup. Seed 42
# is already done (run_no_das_strict / run_no_das).
set -uo pipefail

PY='/c/Users/Zirui Lin/anaconda3/envs/rader_baseline/python.exe'
LOG=/g/zhanghe/output/V9.2.1-Ablation-NoDAS/_chain_seeds_1234_7890.log
mkdir -p "$(dirname "$LOG")"

stage() {
    echo "[$(date '+%H:%M:%S')] === $* ===" | tee -a "$LOG"
}

run_no_das_seed() {
    local DATASET=$1
    local SEED=$2
    local RUN_DIR
    if [ "$DATASET" = "self" ]; then
        RUN_DIR=/g/zhanghe/output/V9.2.1-KnownPeople-UnknownFreq/run_no_das_seed${SEED}_strict
        unset V921_TASK_DIR_NAME V921_RUNS_DIR_TAG V921_EXPERIMENT_NAME
    else
        RUN_DIR=/g/zhanghe/output/V9.2.1-KnownPeople-UnknownFreq-CI4R/run_no_das_seed${SEED}
        export V921_TASK_DIR_NAME=ci4r_known_people_unknown_freq
        export V921_RUNS_DIR_TAG=V9.2.1-KnownPeople-UnknownFreq-CI4R
        export V921_EXPERIMENT_NAME=V9.2.1-KnownPeople-UnknownFreq-CI4R
    fi
    export V921_USE_DAS=0

    if [ -f "${RUN_DIR}/reports/strict_analysis.json" ]; then
        stage "  SKIP: ${DATASET} -DAS seed=${SEED} (already done)"
        return
    fi

    stage "TRAIN: ${DATASET} -DAS seed=${SEED} -> ${RUN_DIR}"
    "$PY" /g/zhanghe/baseline_v9_2_1/train.py \
        --run-dir "$RUN_DIR" \
        --seed "$SEED" \
        --progress simple 2>&1 | tee -a "$LOG"

    stage "ANALYZE: ${DATASET} -DAS seed=${SEED}"
    "$PY" /g/zhanghe/baseline_v9_2_1/strict_analysis.py \
        --run-dir "$RUN_DIR" 2>&1 | tee -a "$LOG"
}

stage "V9.2.1 -DAS multi-seed (1234, 7890) chain START $(date)"
run_no_das_seed self 1234
run_no_das_seed self 7890
run_no_das_seed ci4r 1234
run_no_das_seed ci4r 7890
unset V921_USE_DAS

stage "Aggregating no-DAS x 3 seeds (paired with V9.2.1 full x 3 seeds)..."
"$PY" /g/zhanghe/baseline_v9_2_1/aggregate_ablation_no_das.py 2>&1 | tee -a "$LOG"
stage "V9.2.1 -DAS multi-seed chain FINISHED $(date)"
