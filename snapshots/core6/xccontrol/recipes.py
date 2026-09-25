"""Explicit recipes derived from ABLATION_REPORT M.0-M.8 and RESULTS_SUPP.
These are recovered recipes, not original launch-environment exports.
"""
from __future__ import annotations
from .common import CLASSES6, CLASSES7, FREQUENCIES

BASE = dict(
    EPOCHS=100, SOURCE_BATCH_SIZE=16, NUM_WORKERS=0, IMG_SIZE=224,
    AMP_DTYPE='bf16', DEFAULT_BACKBONE='vit_large_patch16_dinov3.lvd1689m', PRETRAINED=True,
    FEATURE_DROPOUT=0.2, HEAD_HIDDEN=512, ARC_MARGIN=0.25, ARC_SCALE=24.0,
    WARMUP_EPOCHS=3, LR=3e-4, WEIGHT_DECAY=0.05, LABEL_SMOOTHING=0.05,
    SUPCON_WEIGHT=0.25, SUPCON_TEMP=0.1, MIRO_WEIGHT=0.1, LOGIT_ADJUST_TAU=1.0,
    BACKBONE_TUNE_MODE='lora', LORA_RANK=2, LORA_ALPHA=8, LORA_DROPOUT=0.10,
    LORA_TARGET_MODULES=['qkv', 'proj', 'fc1', 'fc2'],
    USE_DANN=False, DANN_WEIGHT=0.1, DANN_HIDDEN=128,
    USE_DAS=True, DAS_MODE='curriculum',
    DAS_STAGE1_END_EPOCH=8, DAS_STAGE1_P=0.35, DAS_STAGE1_F_LOW_GHZ=10.0, DAS_STAGE1_F_HIGH_GHZ=24.0,
    DAS_STAGE2_END_EPOCH=24, DAS_STAGE2_P=0.70, DAS_STAGE2_F_LOW_GHZ=10.0, DAS_STAGE2_F_HIGH_GHZ=50.0,
    DAS_STAGE3_P=1.0, DAS_STAGE3_F_LOW_GHZ=12.0, DAS_STAGE3_F_HIGH_GHZ=95.0,
    DAS_FIXED_P=1.0, DAS_FIXED_FULL_F_LOW_GHZ=15.0, DAS_FIXED_FULL_F_HIGH_GHZ=140.0,
    DAS_FIXED_NARROW_F_LOW_GHZ=10.0, DAS_FIXED_NARROW_F_HIGH_GHZ=30.0,
    DAS_JITTER_P=1.0, DAS_JITTER_RHO_LOW=0.8, DAS_JITTER_RHO_HIGH=1.25,
    USE_FAST_GPU=True, SKIP_ORACLE=False, EVAL_EVERY=1, SKIP_LAST_CKPT=True,
    USE_HFT=True, HFT_P=0.65, HFT_FLOOR_AMP=0.05, HFT_HF_AMP=0.08,
    USE_SPEC_AUGMENT=True, TIME_MASK_P=0.6, DOPPLER_MASK_P=0.6, TIME_MASK_FRAC=0.1, DOPPLER_MASK_FRAC=0.12,
    EMA_DECAY=0.999, EMA_START_EPOCH=5,
    V15_FALSIFY_WEIGHT=0.0, V15_KIN_SOURCE_WEIGHT=0.0, V15_SENSOR_UNIFORM_WEIGHT=0.0,
    V15_CONSIST_WEIGHT=0.0, V15_CONSIST_CONF=0.6, V15_OOD_FREQS=[7.0,77.0,99.0,120.0,140.0],
    V15R_REALISM_WEIGHT=0.0, V15R_WORSTCASE=False, V15R_FALSIFY_WEIGHT=0.0,
    V15R_MARGIN_WEIGHT=0.0, V15R_SINGLE_HEAD=False,
    V13_FREQ_WEIGHT=0.0, V13_DECORR_WEIGHT=0.0, V13_GRL_WEIGHT=0.0,
    V13_GRL_TARGET='shown', V13_GRL_HIDDEN=128, V13_GRL_DISCRETE=False, V13_GRL_BINS=3,
    V16_CONE_WEIGHT=0.0, V17_GRAMMAR_LOSS_WEIGHT=0.0, V18_RECON_WEIGHT=0.0, V18_DOMAIN_WEIGHT=0.0,
    CARRIER_NORM='off', LOWBAND_GHZ=10.0,
)
# Source paper Table II (10 rows) plus four schedule controls from supplementary material.
RECIPES = {
    'no_image_aug': dict(USE_DAS=False, USE_HFT=False, USE_SPEC_AUGMENT=False),
    'radar_aug': dict(USE_DAS=False),
    'jitter': dict(DAS_MODE='jitter'),
    'dann': dict(USE_DAS=False, USE_DANN=True),
    'stretch_only': dict(USE_HFT=False, USE_SPEC_AUGMENT=False),
    'das_only': {},
    'residual_only': dict(V13_FREQ_WEIGHT=0.05),
    'grl_only': dict(V13_GRL_WEIGHT=0.3),
    'discrete_acr': dict(V13_FREQ_WEIGHT=0.05, V13_GRL_WEIGHT=0.3, V13_GRL_DISCRETE=True),
    'proposed': dict(V13_FREQ_WEIGHT=0.05, V13_GRL_WEIGHT=0.3),
    'narrow_das': dict(DAS_MODE='fixed_narrow'),
    'wide_das': dict(DAS_MODE='fixed_full'),
    'narrow_acr': dict(DAS_MODE='fixed_narrow', V13_FREQ_WEIGHT=0.05, V13_GRL_WEIGHT=0.3),
    'wide_acr': dict(DAS_MODE='fixed_full', V13_FREQ_WEIGHT=0.05, V13_GRL_WEIGHT=0.3),
}
EXPECTED_F1_ORIGINAL7 = dict(no_image_aug=0.146, radar_aug=0.302, jitter=0.518, dann=0.275,
    stretch_only=0.704, das_only=0.767, residual_only=0.814, grl_only=0.826,
    discrete_acr=0.796, proposed=0.832, narrow_das=0.804, wide_das=0.756,
    narrow_acr=0.800, wide_acr=0.626)


def make_config(recipe: str, dataset: str, target: str, epochs=100, batch=16, precision='bf16'):
    if recipe not in RECIPES: raise ValueError(recipe)
    out = dict(BASE); out.update(RECIPES[recipe])
    out.update(EPOCHS=epochs, SOURCE_BATCH_SIZE=batch, AMP_DTYPE=precision)
    classes = CLASSES7 if dataset == 'original7' else CLASSES6
    out.update(CLASSES=classes, NUM_CLASSES=len(classes), CLASS_TO_IDX={c:i for i,c in enumerate(classes)},
               DROP_CLASSES=[c for c in ['Away','Bend','Crawl','Kneel','Limp','Pick','SStep','Scissor','Sit','Toes','Towards'] if c not in classes])
    sources = [f for f in FREQUENCIES if f != target]
    out.update(TRAIN_FREQS=sources, TEST_FREQS=[target], FREQ_TO_IDX={f:i for i,f in enumerate(sources)},
               NUM_FREQ_DOMAINS=len(sources))
    return out
