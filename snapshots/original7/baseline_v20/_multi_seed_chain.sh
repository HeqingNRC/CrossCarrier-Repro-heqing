#!/usr/bin/env bash
# Multi-seed V9.2.1 sweep on both self and CI4R datasets.
# Existing seed=42 runs are reused (run_002_strict for self, run_001 for CI4R).
# This chain trains 4 new runs (seeds 1234 and 7890 on each dataset),
# then aggregates biased + strict numbers as mean +/- std across 3 seeds.
set -uo pipefail

PY='/c/Users/Zirui Lin/anaconda3/envs/rader_baseline/python.exe'
LOG=/g/zhanghe/output/V9.2.1-MultiSeed/_chain.log
mkdir -p "$(dirname "$LOG")"

stage() {
    echo "[$(date '+%H:%M:%S')] === $* ===" | tee -a "$LOG"
}

stage "multi-seed V9.2.1 sweep START $(date)"

run_v921() {
    local DATASET=$1     # 'self' or 'ci4r'
    local SEED=$2
    local RUN_DIR
    if [ "$DATASET" = "self" ]; then
        RUN_DIR=/g/zhanghe/output/V9.2.1-KnownPeople-UnknownFreq/run_seed${SEED}_strict
        unset V921_TASK_DIR_NAME
        unset V921_RUNS_DIR_TAG
        unset V921_EXPERIMENT_NAME
    else
        RUN_DIR=/g/zhanghe/output/V9.2.1-KnownPeople-UnknownFreq-CI4R/run_seed${SEED}
        export V921_TASK_DIR_NAME=ci4r_known_people_unknown_freq
        export V921_RUNS_DIR_TAG=V9.2.1-KnownPeople-UnknownFreq-CI4R
        export V921_EXPERIMENT_NAME=V9.2.1-KnownPeople-UnknownFreq-CI4R
    fi

    if [ -f "${RUN_DIR}/reports/strict_analysis.json" ]; then
        stage "  SKIP: ${DATASET} seed=${SEED} (strict_analysis already exists at ${RUN_DIR})"
        return
    fi

    stage "  TRAIN: ${DATASET} seed=${SEED} -> ${RUN_DIR}"
    "$PY" /g/zhanghe/baseline_v9_2_1/train.py \
        --run-dir "$RUN_DIR" \
        --seed "$SEED" \
        --progress simple 2>&1 | tee -a "$LOG"

    stage "  ANALYZE: ${DATASET} seed=${SEED}"
    "$PY" /g/zhanghe/baseline_v9_2_1/strict_analysis.py \
        --run-dir "$RUN_DIR" 2>&1 | tee -a "$LOG"
}

# Existing seed=42 runs (no need to re-train); just verify their strict_analysis.
stage "Stage 0: ensure seed=42 strict_analysis exists for both datasets"
SELF_42=/g/zhanghe/output/V9.2.1-KnownPeople-UnknownFreq/run_002_strict
CI4R_42=/g/zhanghe/output/V9.2.1-KnownPeople-UnknownFreq-CI4R/run_001
[ -f "${SELF_42}/reports/strict_analysis.json" ] || \
    "$PY" /g/zhanghe/baseline_v9_2_1/strict_analysis.py --run-dir "$SELF_42" 2>&1 | tee -a "$LOG"
[ -f "${CI4R_42}/reports/strict_analysis.json" ] || \
    "$PY" /g/zhanghe/baseline_v9_2_1/strict_analysis.py --run-dir "$CI4R_42" 2>&1 | tee -a "$LOG"

stage "Stage 1: train + analyze 4 new runs (seeds 1234, 7890 x self, ci4r)"
run_v921 self 1234
run_v921 self 7890
run_v921 ci4r 1234
run_v921 ci4r 7890
stage "Stage 1 DONE"

stage "Stage 2: aggregate mean +/- std across 3 seeds"
"$PY" /g/zhanghe/baseline_v9_2_1/multi_seed_aggregate.py 2>&1 | tee -a "$LOG"
stage "Stage 2 DONE"

stage "multi-seed V9.2.1 sweep FINISHED $(date)"
