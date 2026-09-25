from pathlib import Path
KIT = Path(__file__).resolve().parent
BASE = Path('/scratch/st-zliu-1/heqingz')
DEFAULT_ASSETS = BASE / 'CrossCarrier_Fresh_20260922/CrossCarrier_Repro'
DEFAULT_PYTHON = BASE / 'envs/xcarrier_20260922/bin/python'
SEEDS = [42, 1234, 31415]
TARGETS = ['10GHz', '24GHz', '77GHz']
RECIPES = ['no_image_aug', 'radar_aug', 'stretch_only', 'das_only',
           'residual_only', 'grl_only', 'discrete_acr', 'proposed', 'wide_acr', 'narrow_acr']
ALIASES = {'no_image_aug': 'DAS_ERM', 'radar_aug': 'E1_noDAS',
           'stretch_only': 'DAS_stretch_only', 'das_only': 'A_REF',
           'residual_only': 'ACR_Lfreq_only', 'grl_only': 'ACR_GRL_only',
           'discrete_acr': 'ACR_discrete', 'proposed': 'A_V13_GRL',
           'wide_acr': 'SCHED_fixedfull_ACR', 'narrow_acr': 'SCHED_fixednarrow_ACR'}
EXPECTED_N = {'10GHz': {'train': 677, 'val': 102, 'test': 73},
              '24GHz': {'train': 496, 'val': 74, 'test': 126},
              '77GHz': {'train': 677, 'val': 102, 'test': 73}}
SPLIT_COUNTS = {'10GHz': {'train':248,'val':37,'test':73},
                '24GHz': {'train':429,'val':65,'test':126},
                '77GHz': {'train':248,'val':37,'test':73}}
def plan_rows():
    return [dict(index=i, dataset='core6', target=t, recipe=r, alias=ALIASES[r], seed=s,
                 epochs=100, batch=16, eval_batch=8, precision='fp16')
            for i, (t,r,s) in enumerate((t,r,s) for t in TARGETS for r in RECIPES for s in SEEDS)]
def smoke_rows():
    return [dict(index=i, target=t, recipe=r, seed=42)
            for i, (t,r) in enumerate([(t,'proposed') for t in TARGETS] +
                                     [('77GHz',r) for r in RECIPES if r!='proposed'])]
