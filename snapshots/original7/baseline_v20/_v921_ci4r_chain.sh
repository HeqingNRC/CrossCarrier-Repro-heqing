#!/usr/bin/env bash
# Train V9.2.1 on CI4R-format manifests, then run strict_analysis.
set -uo pipefail

PY='/c/Users/Zirui Lin/anaconda3/envs/rader_baseline/python.exe'
LOG=/g/zhanghe/output/V9.2.1-KnownPeople-UnknownFreq-CI4R/_chain.log
RUN=/g/zhanghe/output/V9.2.1-KnownPeople-UnknownFreq-CI4R/run_001
mkdir -p "$(dirname "$LOG")"

stage() {
    echo "[$(date '+%H:%M:%S')] === $* ===" | tee -a "$LOG"
}

stage "V9.2.1 on CI4R chain START $(date)"

stage "Stage 1: train V9.2.1 with CI4R manifests"
export V921_TASK_DIR_NAME=ci4r_known_people_unknown_freq
export V921_RUNS_DIR_TAG=V9.2.1-KnownPeople-UnknownFreq-CI4R
export V921_EXPERIMENT_NAME=V9.2.1-KnownPeople-UnknownFreq-CI4R
"$PY" /g/zhanghe/baseline_v9_2_1/train.py \
    --run-dir "$RUN" \
    --progress simple 2>&1 | tee -a "$LOG"
stage "Stage 1 DONE"

stage "Stage 2: strict_analysis on V9.2.1 CI4R run"
"$PY" /g/zhanghe/baseline_v9_2_1/strict_analysis.py --run-dir "$RUN" 2>&1 | tee -a "$LOG"
stage "Stage 2 DONE"

stage "V9.2.1 on CI4R chain FINISHED $(date)"
