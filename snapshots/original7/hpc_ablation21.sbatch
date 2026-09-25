#!/usr/bin/env bash
set -euo pipefail

module load miniconda3

PKG=/scratch/st-zliu-1/heqingz/CrossCarrier_Fresh_20260922/CrossCarrier_Repro
PY=/scratch/st-zliu-1/heqingz/envs/xcarrier_20260922/bin/python
OUT="$PKG/EXPERIMENTSRESULT/REVISION_5090/SUPP_ABLATION"

CONFIGS=(
    ACR_Lfreq_only
    ACR_GRL_only
    ACR_discrete
    DAS_ERM
    DAS_stretch_only
    SCHED_fixedfull_ACR
    SCHED_fixednarrow_ACR
)

SEEDS=(42 1234 31415)

CFG_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % 3))

CFG="${CONFIGS[$CFG_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"

RUN="$OUT/$CFG/seed$SEED"
mkdir -p "$RUN"

cd "$PKG"

# ------------------------------------------------------------
# Bundled DINOv3: force offline local cache
# ------------------------------------------------------------
export HF_HOME="$PKG/weights"
export HUGGINGFACE_HUB_CACHE="$PKG/weights/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ------------------------------------------------------------
# Common recipe for the final DAS/ACR generation
# ------------------------------------------------------------

# V100 does not support native bf16; this is the only deliberate
# hardware adaptation from Zirui's original report.
export V921_AMP_DTYPE=fp16

export V921_SOURCE_BATCH_SIZE=16
export V921_EVAL_BATCH=8
export V921_NUM_WORKERS=0

# Original GPU augmentation path.
export V921_FAST_GPU=1

# Preserve historical HFT RNG behaviour.
export V921_SEEDED_HFT=0

# No DANN in these seven supplementary rows.
export V921_USE_DANN=0

# Old decorrelation is OFF for the final ACR-generation experiments.
export V13_DECORR_WEIGHT=0

# Continuous ACR defaults; individual rows override below.
export V13_GRL_TARGET=shown
export V13_GRL_DISCRETE=0
export V13_GRL_BINS=3

# V15 was dropped in the final DAS+ACR experiment family.
export V15_FALSIFY_WEIGHT=0
export V15_KIN_SOURCE_WEIGHT=0
export V15_SENSOR_UNIFORM_WEIGHT=0
export V15_CONSIST_WEIGHT=0

# Redesigned V15R also OFF in these seven rows.
export V15R_REALISM_WEIGHT=0
export V15R_WORSTCASE=0
export V15R_FALSIFY_WEIGHT=0
export V15R_MARGIN_WEIGHT=0
export V15R_SINGLE_HEAD=0

# Defaults for the seven-row matrix.
export V921_USE_DAS=1
export V921_DAS_MODE=curriculum
export V921_USE_HFT=1
export V921_USE_SPEC_AUGMENT=1

export V13_FREQ_WEIGHT=0
export V13_GRL_WEIGHT=0

case "$CFG" in

    ACR_Lfreq_only)
        # Residual carrier branch only; no adversary.
        export V13_FREQ_WEIGHT=0.05
        export V13_GRL_WEIGHT=0
        ;;

    ACR_GRL_only)
        # Continuous GRL only; no residual supervision.
        export V13_FREQ_WEIGHT=0
        export V13_GRL_WEIGHT=0.3
        ;;

    ACR_discrete)
        # Full ACR, replacing continuous regression adversary with 3-bin CE.
        export V13_FREQ_WEIGHT=0.05
        export V13_GRL_WEIGHT=0.3
        export V13_GRL_DISCRETE=1
        export V13_GRL_BINS=3
        ;;

    DAS_ERM)
        # True no-augmentation ERM.
        export V921_USE_DAS=0
        export V921_USE_HFT=0
        export V921_USE_SPEC_AUGMENT=0
        export V13_FREQ_WEIGHT=0
        export V13_GRL_WEIGHT=0
        ;;

    DAS_stretch_only)
        # Carrier-matched Doppler stretch without generic radar augmentation.
        export V921_USE_DAS=1
        export V921_DAS_MODE=curriculum
        export V921_USE_HFT=0
        export V921_USE_SPEC_AUGMENT=0
        export V13_FREQ_WEIGHT=0
        export V13_GRL_WEIGHT=0
        ;;

    SCHED_fixedfull_ACR)
        # Full ACR with wide fixed DAS from epoch 1.
        export V921_USE_DAS=1
        export V921_DAS_MODE=fixed_full
        export V13_FREQ_WEIGHT=0.05
        export V13_GRL_WEIGHT=0.3
        ;;

    SCHED_fixednarrow_ACR)
        # Full ACR with narrow fixed DAS from epoch 1.
        export V921_USE_DAS=1
        export V921_DAS_MODE=fixed_narrow
        export V13_FREQ_WEIGHT=0.05
        export V13_GRL_WEIGHT=0.3
        ;;

    *)
        echo "Unknown config: $CFG" >&2
        exit 2
        ;;
esac

# Guarantee an ep100 trainable-only EMA checkpoint exists even if
# the historical pool gate does not admit that run.
export DUMP_EPOCH_CKPTS=1

# Avoid the huge resume-only full snapshot; this does not affect training.
export V921_SKIP_LAST_CKPT=1

echo "=================================================="
echo "JOB=$SLURM_JOB_ID"
echo "ARRAY_TASK=$SLURM_ARRAY_TASK_ID"
echo "HOST=$(hostname)"
echo "CFG=$CFG"
echo "SEED=$SEED"
echo "RUN=$RUN"
echo "=================================================="

echo "----- EFFECTIVE ABLATION SWITCHES -----"
env | grep -E \
'^(V921_USE_DAS|V921_DAS_MODE|V921_USE_HFT|V921_USE_SPEC_AUGMENT|V921_USE_DANN|V921_AMP_DTYPE|V13_|V15_|V15R_|DUMP_EPOCH_CKPTS)' \
| sort

"$PY" -u baseline_v20/train.py \
    --epochs 100 \
    --seed "$SEED" \
    --run-dir "$RUN"

echo "DONE CFG=$CFG SEED=$SEED"
