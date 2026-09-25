"""V20 configuration: V15 DAS falsification + V13 carrier residual split.

V9.2.1 keeps the V9.2 task definition and clean dev/test selection protocol,
but gives the backbone a small amount of trainable capacity:
1. default: small LoRA on the frozen DINOv3 feature extractor
2. optional: unfreeze the final transformer block (+ final norm)
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Optional env overrides so V9.2.1 can be re-pointed at e.g. a CI4R-format
# manifest tree without touching the rest of the codebase.
_TASK_DIR_NAME = os.environ.get("V921_TASK_DIR_NAME", "known_people_unknown_freq")
TASK_DIR = ROOT / "tasks" / _TASK_DIR_NAME
MANIFEST_DIR = TASK_DIR / "manifest"
DATASET_ROOT = ROOT
RUNS_DIR = ROOT / "output" / os.environ.get(
    "V921_RUNS_DIR_TAG", "V20-V15V13-KnownPeople-UnknownFreq",
)
WEIGHTS_DIR = ROOT / "weights"

EXPERIMENT_NAME = os.environ.get("V921_EXPERIMENT_NAME",
                                  "V20-V15V13-KnownPeople-UnknownFreq")

SOURCE_CLASSES = [
    "Away", "Bend", "Crawl", "Kneel", "Limp", "Pick",
    "SStep", "Scissor", "Sit", "Toes", "Towards",
]
DROP_CLASSES = ["Crawl", "Limp", "Scissor", "Toes"]
CLASSES = ["Away", "Bend", "Kneel", "Pick", "SStep", "Sit", "Towards"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

TRAIN_FREQS = ["10GHz", "24GHz"]
TEST_FREQS = ["77GHz"]
FREQ_TO_IDX = {f: i for i, f in enumerate(TRAIN_FREQS)}
NUM_FREQ_DOMAINS = len(TRAIN_FREQS)

# Deterministic split inside the original 77GHz benchmark.
# Each class contributes exactly DEV_PER_CLASS images to the dev split, chosen
# in a subject-aware proportional way. The rest stays untouched as final test.
TEST77_DEV_PER_CLASS = 20
TEST77_DEV_SPLIT_SEED = 42

IMG_SIZE = 224
SOURCE_BATCH_SIZE = int(os.environ.get("V921_SOURCE_BATCH_SIZE", "16"))
PER_FREQ_BATCH = 8
NUM_WORKERS = int(os.environ.get("V921_NUM_WORKERS", "0"))
SEED = 42
AMP_DTYPE = "bf16"

DEFAULT_BACKBONE = "vit_large_patch16_dinov3.lvd1689m"

PRETRAINED = True
FEATURE_DROPOUT = 0.2
HEAD_HIDDEN = 512
ARC_MARGIN = float(os.environ.get("V921_ARC_MARGIN", "0.25"))   # set 0 for plain (no-ArcFace) baseline
ARC_SCALE = 24.0

EPOCHS = 80
WARMUP_EPOCHS = 3
LR = 3e-4
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.05

SUPCON_WEIGHT = float(os.environ.get("V921_SUPCON_WEIGHT", "0.25"))   # 0 = off (baseline)
MIRO_WEIGHT = float(os.environ.get("V921_MIRO_WEIGHT", "0.1"))         # 0 = off (baseline)
LOGIT_ADJUST_TAU = 1.0

# Backbone tuning mode for V9.2.1.
# `lora` is the default experimental path. `last_block` is kept as a direct
# ablation without changing the rest of the V9.2 recipe.
BACKBONE_TUNE_MODE = "lora"
LORA_RANK = 2
LORA_ALPHA = 8
LORA_DROPOUT = 0.10
LORA_TARGET_MODULES = ["qkv", "proj", "fc1", "fc2"]
LAST_BLOCK_UNFREEZE_NORM = True

# V9.2.1 keeps the V9.1/V9.2 no-DANN ablation setting.
# Set V921_USE_DANN=1 to enable a band-adversarial DANN head on the source
# 10/24 GHz batch (binary domain classification through a gradient reversal
# layer). Used as a head-to-head comparison against the DAS curriculum:
# DAS = physics-driven cross-band mechanism, DANN = data-driven alternative.
USE_DANN = os.environ.get("V921_USE_DANN", "0") == "1"
DANN_WEIGHT = float(os.environ.get("V921_DANN_WEIGHT", "0.1"))
DANN_HIDDEN = 128

SUPCON_TEMP = 0.1

# Same DAS curriculum as V9/V9.2.
# Set V921_USE_DAS=0 to disable DAS curriculum (ablation experiment).
USE_DAS = os.environ.get("V921_USE_DAS", "1") == "1"
DAS_STAGE1_END_EPOCH = 8
DAS_STAGE1_P = 0.35
DAS_STAGE1_F_LOW_GHZ = 10.0
DAS_STAGE1_F_HIGH_GHZ = 24.0

DAS_STAGE2_END_EPOCH = 24
DAS_STAGE2_P = 0.70
DAS_STAGE2_F_LOW_GHZ = 10.0
DAS_STAGE2_F_HIGH_GHZ = 50.0

DAS_STAGE3_P = 1.0
DAS_STAGE3_F_LOW_GHZ = 12.0
DAS_STAGE3_F_HIGH_GHZ = 95.0

# DAS schedule ablation. `curriculum` is the default V9.2.1 behaviour
# (3-stage warmup -> mid -> wide). `fixed_full` and `fixed_narrow` apply
# DAS-as-operator throughout training (no curriculum), to test whether
# DAS's value comes from the schedule itself or from the operator.
#
#   curriculum    -> existing 3-stage schedule (default, no behaviour change)
#   fixed_full    -> [15, 140] GHz with p=1.0 from epoch 1 (matches curriculum
#                    final-stage range exactly; only the schedule differs)
#   fixed_narrow  -> [10, 30] GHz with p=1.0 from epoch 1 (Kern 2022 style,
#                    small extrapolation: 30/24 GHz ~= 1.25x source)
#   jitter        -> physics-free Doppler-axis jitter (Experiment 1 control):
#                    rescale ratio rho ~ U[RHO_LOW, RHO_HIGH] drawn DIRECTLY,
#                    decoupled from any carrier ratio. Same operator/budget as
#                    DAS but no f_d ∝ f_c physics, so it isolates "is the gain
#                    from carrier-matched DAS or from generic axis scaling?".
DAS_MODE = os.environ.get("V921_DAS_MODE", "curriculum")
DAS_FIXED_P = 1.0
DAS_FIXED_FULL_F_LOW_GHZ = 15.0
DAS_FIXED_FULL_F_HIGH_GHZ = 140.0
DAS_FIXED_NARROW_F_LOW_GHZ = 10.0
DAS_FIXED_NARROW_F_HIGH_GHZ = 30.0

# Physics-free jitter control (only used when DAS_MODE='jitter').
DAS_JITTER_P = float(os.environ.get("V921_DAS_JITTER_P", "1.0"))
DAS_JITTER_RHO_LOW = float(os.environ.get("V921_DAS_JITTER_RHO_LOW", "0.8"))
DAS_JITTER_RHO_HIGH = float(os.environ.get("V921_DAS_JITTER_RHO_HIGH", "1.25"))

# ---- GPU acceleration (see G:\zhanghe\GPU_TRAINING_SPEEDUP_PLAYBOOK.md) ----
# FAST_GPU: cache images in VRAM + run DAS/HFT/standardize/SpecAugment on GPU for
#   training, and serve cached big-batch eval. Default OFF = original CPU path.
#   NOTE: GPU DAS uses grid_sample (not bit-identical to PIL das_deterministic) ->
#   absolute numbers may drift slightly; validate a known headline before trusting.
USE_FAST_GPU = os.environ.get("V921_FAST_GPU", "0") == "1"
# SKIP_ORACLE: skip the frozen oracle forward on x_aux (training) -- oracle output
#   is only used by MIRO on x_src, so on x_aux it is pure waste. Bit-identical.
SKIP_ORACLE = os.environ.get("V921_SKIP_ORACLE", "0") == "1"
EVAL_BATCH = int(os.environ.get("V921_EVAL_BATCH", "256"))
# Run the per-epoch DIAGNOSTIC eval (val/test, live+EMA) only every EVAL_EVERY
# epochs (ep1, the last epoch, and every POOL_PERIOD are always evaluated). 1 =
# every epoch (current behaviour, fully inert). Higher skips the diagnostic eval
# on most epochs: ~4 ViT-L eval passes/epoch saved (~20-25% wall-clock). Does NOT
# affect the final-EMA pool_ep100_ema.pt result -- pool epochs stay evaluated --
# only history.json granularity. See GPU_TRAINING_SPEEDUP_PLAYBOOK.md.
EVAL_EVERY = int(os.environ.get("V921_EVAL_EVERY", "1"))
# Skip the per-epoch ~2.4GB last.pt snapshot (resume-only; NOT used by training or
# checkpoint selection). Removes disk I/O; does NOT affect results or metrics.
SKIP_LAST_CKPT = os.environ.get("V921_SKIP_LAST_CKPT", "0") == "1"

# HFT / SpecAugment are generic radar augmentations applied INDEPENDENTLY of the
# DAS Doppler-stretch (both default ON -> reproduces current behaviour exactly).
# Env-gated for the P0-B "DAS body" ablation: a true ERM row needs all three off
# (USE_DAS=0, USE_HFT=0, USE_SPEC_AUGMENT=0); a stretch-only row needs USE_DAS=1
# with USE_HFT=0, USE_SPEC_AUGMENT=0. Honored on both CPU and FAST_GPU paths.
USE_HFT = os.environ.get("V921_USE_HFT", "1") == "1"
HFT_P = 0.65
HFT_FLOOR_AMP = 0.05
HFT_HF_AMP = 0.08

USE_SPEC_AUGMENT = os.environ.get("V921_USE_SPEC_AUGMENT", "1") == "1"
TIME_MASK_P = 0.6
DOPPLER_MASK_P = 0.6
TIME_MASK_FRAC = 0.10
DOPPLER_MASK_FRAC = 0.12

EMA_DECAY = 0.999
EMA_START_EPOCH = 5

FREQ_SCAN_POINTS = [15.0, 30.0, 50.0, 77.0, 99.0, 120.0, 140.0]

LOG_EVERY_STEPS = 10
TERMINAL_PROGRESS = "simple"

# Source-qualified candidate policy used by V20 experiments.
# The user protocol is: once a checkpoint reaches >=90% source validation
# performance, it enters the candidate pool; the diagnostic winner is then the
# candidate with the strongest 77GHz final-test macro-F1.
SOURCE_QUALIFIED_MIN_F1 = float(os.environ.get("SOURCE_QUALIFIED_MIN_F1", "0.90"))
SOURCE_QUALIFIED_MIN_ACC = float(
    os.environ.get("SOURCE_QUALIFIED_MIN_ACC", os.environ.get("SOURCE_QUALIFIED_MIN_F1", "0.90"))
)
SOURCE_QUALIFIED_METRIC = os.environ.get("SOURCE_QUALIFIED_METRIC", "acc")
SOURCE_QUALIFIED_BASIS = os.environ.get("SOURCE_QUALIFIED_BASIS", "ema")

# V15: DAS falsification risk minimization.
# Out-of-source DAS views are treated as stress tests: the kinematic head is
# trained to stay correct, while the sensor head is discouraged from making
# confident predictions on those extrapolated views.
V15_FALSIFY_WEIGHT = float(os.environ.get("V15_FALSIFY_WEIGHT", "0.20"))
V15_KIN_SOURCE_WEIGHT = float(os.environ.get("V15_KIN_SOURCE_WEIGHT", "0.35"))
V15_SENSOR_UNIFORM_WEIGHT = float(os.environ.get("V15_SENSOR_UNIFORM_WEIGHT", "0.05"))
V15_CONSIST_WEIGHT = float(os.environ.get("V15_CONSIST_WEIGHT", "0.08"))
V15_CONSIST_CONF = float(os.environ.get("V15_CONSIST_CONF", "0.60"))
V15_OOD_FREQS = [7.0, 77.0, 99.0, 120.0, 140.0]

# V15R: redesigned falsification (stacks on the GRL residual split). All default
# OFF -> inert (reproduces A_V13_GRL). Three independent pieces:
#  (a) realism: carrier-scaled high-freq Doppler texture. gpu_hft HF amplitude is
#      multiplied per-sample by (f_virt / V15R_HFT_REF_GHZ) ** V15R_REALISM_WEIGHT
#      so higher virtual carriers get stronger/finer micro-Doppler texture (closes
#      the real-77 vs DAS-77 gap). 0 = plain fixed-amplitude HFT (current).
#  (b) worst-case carrier + hardest-negative margin on the KINEMATIC head:
#      V15R_WORSTCASE=1 picks, per sample, the candidate carrier (V15R_WC_FREQS)
#      that MAXIMIZES the kinematic CE, then we minimize on it. V15R_FALSIFY_WEIGHT
#      weights that worst-view CE; V15R_MARGIN_WEIGHT weights a hardest-negative
#      margin loss (true-class logit >= hardest-negative logit + V15R_MARGIN, raw
#      ARC_SCALE units). Replaces the old conf-gated KL consistency.
#  (c) single head: V15R_SINGLE_HEAD=1 trains CE on the kinematic head only (the
#      sensor head is a verified no-op); also set V15_SENSOR_UNIFORM_WEIGHT=0.
V15R_REALISM_WEIGHT = float(os.environ.get("V15R_REALISM_WEIGHT", "0.0"))   # gamma; 0=off
V15R_HFT_REF_GHZ = float(os.environ.get("V15R_HFT_REF_GHZ", "24.0"))
V15R_WORSTCASE = os.environ.get("V15R_WORSTCASE", "0") == "1"
V15R_FALSIFY_WEIGHT = float(os.environ.get("V15R_FALSIFY_WEIGHT", "0.0"))   # worst-view CE
V15R_MARGIN_WEIGHT = float(os.environ.get("V15R_MARGIN_WEIGHT", "0.0"))     # hardest-neg margin
V15R_MARGIN = float(os.environ.get("V15R_MARGIN", "4.0"))                   # ARC_SCALE logit units
V15R_SINGLE_HEAD = os.environ.get("V15R_SINGLE_HEAD", "0") == "1"
V15R_WC_FREQS = [float(x) for x in os.environ.get(
    "V15R_WC_FREQS", "7,77,99,120,140").split(",") if x.strip()]

# V13: carrier residual representation split at the neck.
V13_FREQ_WEIGHT = float(os.environ.get("V13_FREQ_WEIGHT", "0.05"))
V13_DECORR_WEIGHT = float(os.environ.get("V13_DECORR_WEIGHT", "0.01"))

# V13-as-GRL: adversarial carrier-invariance head on z_cls. When V13_GRL_WEIGHT>0
# the weak covariance decorrelation is REPLACED by a Gradient-Reversal-Layer
# adversary that regresses the CONTINUOUS log-carrier from z_cls; through the
# reversed gradient the encoder is pushed to make z_cls carrier-uninformative.
# Default 0 = OFF (existing runs unaffected). Run with V13_DECORR_WEIGHT=0 and
# keep V13_FREQ_WEIGHT=0.05 so z_freq still absorbs the carrier.
V13_GRL_WEIGHT = float(os.environ.get("V13_GRL_WEIGHT", "0.0"))   # 0=off
V13_GRL_HIDDEN = int(os.environ.get("V13_GRL_HIDDEN", "128"))
# What carrier the adversary regresses (and therefore what z_cls is scrubbed of):
#   "shown" -> log(f_src)+r_src  = the EFFECTIVE carrier the (DAS'd) sample shows.
#              Because DAS reaches p=1.0 with f_virt drawn independently of the
#              base band, this DECOUPLES from the base 10-vs-24 carrier, so the
#              base-carrier probe does not move (diagnosed 2026-06-08).
#   "base"  -> log(f_src) only   = the ORIGINAL acquisition band. Forces z_cls to
#              be unable to tell 10 vs 24 GHz even through DAS distortion, i.e.
#              directly targets what the carrier-leakage probe measures.
V13_GRL_TARGET = os.environ.get("V13_GRL_TARGET", "shown")   # shown | base

# V13-as-GRL DISCRETE mode (P0-A "ordinary discrete domain adversary" control).
# Default 0 = OFF -> the adversary is our CONTINUOUS log-carrier regressor (1 output,
# SmoothL1 on log f_eff); existing runs are byte-identical. When V13_GRL_DISCRETE=1
# the adversary instead outputs V13_GRL_BINS logits and is trained with CROSS-ENTROPY
# on a carrier-BIN label: the continuous log f_eff (= log f_src + r, the same target
# the regressor sees) is bucketed into V13_GRL_BINS equal-width bins over the DAS
# curriculum log-carrier range (see v9_2_1lib.das_log_carrier_range; the bin edges
# are computed once at trainer setup and logged). This is the standard discrete
# domain-adversary baseline vs our continuous-carrier regressor. Same grad_reverse
# (z_cls, lambda) and same V13_GRL_WEIGHT as the continuous mode.
V13_GRL_DISCRETE = os.environ.get("V13_GRL_DISCRETE", "0") == "1"
V13_GRL_BINS = int(os.environ.get("V13_GRL_BINS", "3"))

# ---- Reviewer supplementary experiments (2026-06-08) ----
# Family A (#1) -- deterministic physical baseline. Resample every image's
# Doppler axis to a COMMON reference carrier (das scale = ref / f_src), applied
# to BOTH train and eval. This is the method's own das operator used
# deterministically by the KNOWN carrier, i.e. "normalize Doppler to a
# carrier-invariant velocity axis". Run with V921_USE_DAS=0 and V13/V15 weights
# = 0 to isolate "is a simple physical normalization already enough?".
#   off  -> normal pipeline (no normalization)
#   all  -> normalize every band to CARRIER_NORM_REF_GHZ
# Reference-carrier choices used for the ablation:
#   A1 velocity-axis (balanced)   : 27.75  (= sqrt(10*77), min worst-case |log rho|)
#   A2 deploy-carrier             : 77.0   (test stays native scale=1, train -> 77)
#   A3 pad-only lowest            : 10.0   (compress only, never crops train content)
CARRIER_NORM = os.environ.get("V921_CARRIER_NORM", "off")            # off | all
CARRIER_NORM_REF_GHZ = float(os.environ.get("V921_CARRIER_NORM_REF_GHZ", "27.75"))

# Family B (#2) -- carrier-ratio sensitivity. Override the nominal 10 GHz
# low-band SOURCE carrier with its true impulse-UWB effective centre (~7.3 GHz)
# so the DAS physics f_d ∝ f_c uses the corrected carrier. Only the 10 GHz band
# is affected; 24/77 GHz unchanged; the 77 GHz test is never DAS'd.
LOWBAND_GHZ = float(os.environ.get("V921_LOWBAND_GHZ", "10.0"))
