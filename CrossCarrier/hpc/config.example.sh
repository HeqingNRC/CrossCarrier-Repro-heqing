#!/usr/bin/env bash
# Copy to config.sh and replace every REPLACE_ME value with your Sockeye details.
# This file is sourced by Bash. Quote paths; do not put passwords/tokens here.

SOCKEYE_ACCOUNT='REPLACE_ME_ALLOCATION_ACCOUNT'
SOCKEYE_PARTITION='REPLACE_ME_GPU_PARTITION'
PROJECT_DIR='/REPLACE_ME_PROJECT_PATH/CrossCarrier_Repro'
OUTPUT_ROOT='/REPLACE_ME_SCRATCH_OR_PROJECT_PATH/crosscarrier-runs'

# An existing Bash file that loads the chosen modules and activates your Linux
# environment. Resolve actual module names/versions on Sockeye before writing it.
ENV_SETUP='/REPLACE_ME_PROJECT_PATH/activate-crosscarrier.sh'

# Starting resource request, not an assertion about Sockeye partition limits.
# Check `sinfo` and `scontrol show partition` and adjust before submission.
CPUS_PER_TASK=4
MEMORY='32G'
REPRO_WALLTIME='01:00:00'
TRAIN_WALLTIME='06:00:00'
PREFLIGHT_WALLTIME='00:10:00'

# Array: 3 target frequencies x 3 seeds. Limit concurrent jobs to 1 by default.
MAX_CONCURRENT=1
EPOCHS=100
VARIANT='proposed'
PRECISION='fp16'
REPRO_BATCH_SIZE=8
TRAIN_BATCH_SIZE=16
EVAL_BATCH_SIZE=8
NUM_WORKERS=4
# Set EPOCHS=1 and TRAIN_MAX_STEPS=8 only for a labeled training smoke test.
# Empty means full epochs. Clear this again before the final experiment.
TRAIN_MAX_STEPS=''
N_BOOT=2000
PYTHON_BIN='python'
