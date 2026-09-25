#!/usr/bin/env bash
set -euo pipefail

PKG=/scratch/st-zliu-1/heqingz/CrossCarrier_Fresh_20260922/CrossCarrier_Repro
PY=/scratch/st-zliu-1/heqingz/envs/xcarrier_20260922/bin/python
OUT=/scratch/st-zliu-1/heqingz/CrossCarrier_Anchor_Runs_20260924

CONFIGS=(
    A_V13_GRL
    A_REF
    E1_noDAS
)

SEEDS=(42 1234 31415)

CFG_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % 3))

CFG="${CONFIGS[$CFG_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"
RUN="$OUT/$CFG/seed$SEED"

cd "$PKG"

# ------------------------------------------------------------
# Fixed environment / original7 protocol
# ------------------------------------------------------------

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

export HF_HOME="$PKG/weights"
export HUGGINGFACE_HUB_CACHE="$PKG/weights/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export V921_TASK_DIR_NAME=known_people_unknown_freq
export V921_TASK_ROOT="$PKG/tasks/known_people_unknown_freq"
export V921_DATASET_ROOT="$PKG"

export V921_TRAIN_FREQS="10GHz,24GHz"
export V921_TEST_FREQS="77GHz"

# V100
export V921_AMP_DTYPE=fp16
export V921_FAST_GPU=1
export V921_SOURCE_BATCH_SIZE=16
export V921_NUM_WORKERS=0
export V921_EVAL_BATCH=8

# Keep identical generic recipe across the three anchors
export V921_USE_HFT=1
export V921_SEEDED_HFT=0
export V921_USE_SPEC_AUGMENT=1
export V921_USE_DANN=0

export V921_DAS_MODE=curriculum

# Final-method generation: old decorrelation and V15 family OFF
export V13_DECORR_WEIGHT=0
export V13_GRL_TARGET=shown
export V13_GRL_DISCRETE=0
export V13_GRL_BINS=3

export V15_FALSIFY_WEIGHT=0
export V15_KIN_SOURCE_WEIGHT=0
export V15_SENSOR_UNIFORM_WEIGHT=0
export V15_CONSIST_WEIGHT=0

export V15R_REALISM_WEIGHT=0
export V15R_WORSTCASE=0
export V15R_FALSIFY_WEIGHT=0
export V15R_MARGIN_WEIGHT=0
export V15R_SINGLE_HEAD=0

export V16_CONE_WEIGHT=0
export V17_GRAMMAR_LOSS_WEIGHT=0
export V18_RECON_WEIGHT=0
export V18_DOMAIN_WEIGHT=0

export V921_CARRIER_NORM=off
export V921_LOWBAND_GHZ=10.0

# Always preserve ep100 EMA even if source-val pool gate fails
export DUMP_EPOCH_CKPTS=1
export V921_SKIP_LAST_CKPT=1

# ------------------------------------------------------------
# Three missing anchors
# ------------------------------------------------------------

case "$CFG" in

    A_V13_GRL)
        # Full proposed = full DAS + residual branch + continuous GRL
        export V921_USE_DAS=1
        export V13_FREQ_WEIGHT=0.05
        export V13_GRL_WEIGHT=0.30
        ;;

    A_REF)
        # Full DAS only = stretch + HFT + SpecAug, no ACR
        export V921_USE_DAS=1
        export V13_FREQ_WEIGHT=0
        export V13_GRL_WEIGHT=0
        ;;

    E1_noDAS)
        # Historical radar-augmentation control:
        # NO carrier-matched Doppler stretch,
        # HFT + SpecAug remain ON.
        export V921_USE_DAS=0
        export V13_FREQ_WEIGHT=0
        export V13_GRL_WEIGHT=0
        ;;

    *)
        echo "Unknown CFG=$CFG"
        exit 2
        ;;
esac

if [[ -e "$RUN" ]]; then
    echo "REFUSING TO OVERWRITE EXISTING RUN: $RUN"
    exit 3
fi

mkdir -p "$RUN"

echo "=================================================="
echo "ANCHOR FRESH TRAINING"
echo "JOB=$SLURM_JOB_ID"
echo "TASK=$SLURM_ARRAY_TASK_ID"
echo "CFG=$CFG"
echo "SEED=$SEED"
echo "RUN=$RUN"
echo "HOST=$(hostname)"
echo "=================================================="

echo "----- EFFECTIVE RECIPE -----"
env | grep -E \
'^(V921_|V13_|V15_|V15R_|V16_|V17_|V18_|DUMP_EPOCH_CKPTS)' \
| sort

"$PY" -u baseline_v20/train.py \
    --epochs 100 \
    --seed "$SEED" \
    --run-dir "$RUN"

test -f "$RUN/epoch_ckpts/ep100.pt"

echo
echo "TRAIN_COMPLETE CFG=$CFG SEED=$SEED"
