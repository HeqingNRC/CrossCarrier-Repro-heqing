#!/usr/bin/env bash
# Ablation: V9.2.1 with the 10/24 GHz pair-consistency loss disabled.
# Runs the SAME pipeline as V9.2.1 full but:
#   USE_PAIR=0  (force pair_scale=0; pair forward still runs but contributes
#                no gradient -- equivalent to dropping L_pair entirely)
#   USE_DAS=1   (default; DAS curriculum kept on)
#   USE_DANN=0  (default; no domain-adversarial head)
#
# Used to isolate the contribution of pair-consistency vs DAS curriculum.
# 3 seeds (42, 1234, 7890) x 2 datasets (self, ci4r) = 6 runs.
set -uo pipefail

PY='/c/Users/Zirui Lin/anaconda3/envs/rader_baseline/python.exe'
LOG=/g/zhanghe/output/V9.2.1-Ablation-NoPair/_chain.log
mkdir -p "$(dirname "$LOG")"

stage() {
    echo "[$(date '+%H:%M:%S')] === $* ===" | tee -a "$LOG"
}

run_no_pair_seed() {
    local DATASET=$1
    local SEED=$2
    local RUN_DIR
    if [ "$DATASET" = "self" ]; then
        RUN_DIR=/g/zhanghe/output/V9.2.1-KnownPeople-UnknownFreq/run_no_pair_seed${SEED}_strict
        unset V921_TASK_DIR_NAME V921_RUNS_DIR_TAG V921_EXPERIMENT_NAME
    else
        RUN_DIR=/g/zhanghe/output/V9.2.1-KnownPeople-UnknownFreq-CI4R/run_no_pair_seed${SEED}
        export V921_TASK_DIR_NAME=ci4r_known_people_unknown_freq
        export V921_RUNS_DIR_TAG=V9.2.1-KnownPeople-UnknownFreq-CI4R
        export V921_EXPERIMENT_NAME=V9.2.1-KnownPeople-UnknownFreq-CI4R
    fi
    export V921_USE_PAIR=0
    export V921_USE_DAS=1
    export V921_USE_DANN=0

    if [ -f "${RUN_DIR}/reports/strict_analysis.json" ]; then
        stage "  SKIP: ${DATASET} -pair seed=${SEED} (already done)"
        return
    fi

    stage "TRAIN: ${DATASET} -pair seed=${SEED} -> ${RUN_DIR}"
    "$PY" /g/zhanghe/baseline_v9_2_1/train.py \
        --run-dir "$RUN_DIR" \
        --seed "$SEED" \
        --progress simple 2>&1 | tee -a "$LOG"

    stage "ANALYZE: ${DATASET} -pair seed=${SEED}"
    "$PY" /g/zhanghe/baseline_v9_2_1/strict_analysis.py \
        --run-dir "$RUN_DIR" 2>&1 | tee -a "$LOG"
}

stage "V9.2.1 -pair ablation chain START $(date)"

run_no_pair_seed self 42
run_no_pair_seed self 1234
run_no_pair_seed self 7890
run_no_pair_seed ci4r 42
run_no_pair_seed ci4r 1234
run_no_pair_seed ci4r 7890

unset V921_USE_PAIR V921_USE_DAS V921_USE_DANN

stage "Aggregating -pair x 3 seeds against V9.2.1 full and -DAS..."
"$PY" /g/zhanghe/baseline_v9_2_1/aggregate_ablation_no_pair.py 2>&1 | tee -a "$LOG"
stage "V9.2.1 -pair ablation chain FINISHED $(date)"
