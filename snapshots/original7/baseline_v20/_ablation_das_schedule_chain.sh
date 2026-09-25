#!/usr/bin/env bash
# Ablation: DAS schedule. Same V9.2.1 backbone, training recipe, selection
# protocol, and seed set as the full curriculum runs and the -DAS runs.
# Two new configurations:
#   fixed_full    -> DAS-as-operator at [15, 140] GHz, p=1.0 from epoch 1
#                    (matches curriculum's final stage range exactly --
#                    only the schedule itself differs)
#   fixed_narrow  -> DAS-as-operator at [10, 30] GHz, p=1.0 from epoch 1
#                    (Kern 2022 style small extrapolation, 30/24 ~= 1.25x)
#
# Together with the existing curriculum and -DAS runs, this gives a 4-rung
# DAS schedule ladder paired across 3 seeds x 2 datasets.
#
# 2 modes x 3 seeds x 2 datasets = 12 new runs.
set -uo pipefail

PY='/c/Users/Zirui Lin/anaconda3/envs/rader_baseline/python.exe'
LOG=/g/zhanghe/output/V9.2.1-Ablation-DAS-Schedule/_chain.log
mkdir -p "$(dirname "$LOG")"

stage() {
    echo "[$(date '+%H:%M:%S')] === $* ===" | tee -a "$LOG"
}

run_das_seed() {
    local DATASET=$1
    local MODE=$2
    local SEED=$3
    local TAG
    if [ "$MODE" = "fixed_full" ]; then
        TAG="das_fixed_full"
    elif [ "$MODE" = "fixed_narrow" ]; then
        TAG="das_fixed_narrow"
    else
        echo "BAD MODE: $MODE" >&2
        return 1
    fi
    local RUN_DIR
    if [ "$DATASET" = "self" ]; then
        RUN_DIR=/g/zhanghe/output/V9.2.1-KnownPeople-UnknownFreq/run_${TAG}_seed${SEED}_strict
        unset V921_TASK_DIR_NAME V921_RUNS_DIR_TAG V921_EXPERIMENT_NAME
    else
        RUN_DIR=/g/zhanghe/output/V9.2.1-KnownPeople-UnknownFreq-CI4R/run_${TAG}_seed${SEED}
        export V921_TASK_DIR_NAME=ci4r_known_people_unknown_freq
        export V921_RUNS_DIR_TAG=V9.2.1-KnownPeople-UnknownFreq-CI4R
        export V921_EXPERIMENT_NAME=V9.2.1-KnownPeople-UnknownFreq-CI4R
    fi

    # DAS schedule ablation: USE_DAS=1, DAS_MODE=$MODE, DANN off.
    export V921_USE_DAS=1
    export V921_USE_DANN=0
    export V921_DAS_MODE=$MODE

    if [ -f "${RUN_DIR}/reports/strict_analysis.json" ]; then
        stage "  SKIP: ${DATASET} ${MODE} seed=${SEED} (already done)"
        return
    fi

    stage "TRAIN: ${DATASET} DAS_MODE=${MODE} seed=${SEED} -> ${RUN_DIR}"
    "$PY" /g/zhanghe/baseline_v9_2_1/train.py \
        --run-dir "$RUN_DIR" \
        --seed "$SEED" \
        --progress simple 2>&1 | tee -a "$LOG"

    stage "ANALYZE: ${DATASET} DAS_MODE=${MODE} seed=${SEED}"
    "$PY" /g/zhanghe/baseline_v9_2_1/strict_analysis.py \
        --run-dir "$RUN_DIR" 2>&1 | tee -a "$LOG"
}

stage "V9.2.1 DAS schedule ablation chain START $(date)"

# self dataset first (faster: ~22 min/run x 6 = ~2.2 hr)
run_das_seed self fixed_full   42
run_das_seed self fixed_full   1234
run_das_seed self fixed_full   7890
run_das_seed self fixed_narrow 42
run_das_seed self fixed_narrow 1234
run_das_seed self fixed_narrow 7890

# ci4r dataset (~50 min/run x 6 = ~5 hr)
run_das_seed ci4r fixed_full   42
run_das_seed ci4r fixed_full   1234
run_das_seed ci4r fixed_full   7890
run_das_seed ci4r fixed_narrow 42
run_das_seed ci4r fixed_narrow 1234
run_das_seed ci4r fixed_narrow 7890

unset V921_USE_DAS V921_USE_DANN V921_DAS_MODE \
      V921_TASK_DIR_NAME V921_RUNS_DIR_TAG V921_EXPERIMENT_NAME

stage "Aggregating DAS schedule ladder (curriculum / fixed_full / fixed_narrow / -DAS)..."
"$PY" /g/zhanghe/baseline_v9_2_1/aggregate_ablation_das_schedule.py 2>&1 | tee -a "$LOG"
stage "V9.2.1 DAS schedule ablation chain FINISHED $(date)"
