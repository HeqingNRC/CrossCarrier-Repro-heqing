#!/usr/bin/env bash
# Ablation: V9.2.1 - DAS curriculum (USE_DAS=False) on both datasets.
# Single seed (42) for both runs to keep within ~2h total.
set -uo pipefail

PY='/c/Users/Zirui Lin/anaconda3/envs/rader_baseline/python.exe'
LOG=/g/zhanghe/output/V9.2.1-Ablation-NoDAS/_chain.log
mkdir -p "$(dirname "$LOG")"

stage() {
    echo "[$(date '+%H:%M:%S')] === $* ===" | tee -a "$LOG"
}

run_no_das() {
    local DATASET=$1
    local RUN_DIR
    if [ "$DATASET" = "self" ]; then
        RUN_DIR=/g/zhanghe/output/V9.2.1-KnownPeople-UnknownFreq/run_no_das_strict
        unset V921_TASK_DIR_NAME
        unset V921_RUNS_DIR_TAG
        unset V921_EXPERIMENT_NAME
    else
        RUN_DIR=/g/zhanghe/output/V9.2.1-KnownPeople-UnknownFreq-CI4R/run_no_das
        export V921_TASK_DIR_NAME=ci4r_known_people_unknown_freq
        export V921_RUNS_DIR_TAG=V9.2.1-KnownPeople-UnknownFreq-CI4R
        export V921_EXPERIMENT_NAME=V9.2.1-KnownPeople-UnknownFreq-CI4R
    fi
    export V921_USE_DAS=0          # <<< the ablation lever

    stage "TRAIN: ${DATASET} no-DAS -> ${RUN_DIR}"
    "$PY" /g/zhanghe/baseline_v9_2_1/train.py \
        --run-dir "$RUN_DIR" \
        --seed 42 \
        --progress simple 2>&1 | tee -a "$LOG"

    stage "ANALYZE: ${DATASET} no-DAS"
    "$PY" /g/zhanghe/baseline_v9_2_1/strict_analysis.py \
        --run-dir "$RUN_DIR" 2>&1 | tee -a "$LOG"
}

stage "V9.2.1 -DAS ablation chain START $(date)"
run_no_das self
run_no_das ci4r
unset V921_USE_DAS
stage "V9.2.1 -DAS ablation chain FINISHED $(date)"
