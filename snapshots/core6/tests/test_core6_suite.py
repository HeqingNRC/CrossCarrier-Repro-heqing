#!/usr/bin/env python3
"""Contract tests; CPU only. Does not require radar data or a real DINOv3 model."""
import ast,collections,json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from suite_constants import KIT, plan_rows, smoke_rows, RECIPES, SEEDS, TARGETS, EXPECTED_N, SPLIT_COUNTS
from xccontrol.common import CLASSES6,read_csv,verify_source
from xccontrol.recipes import make_config

class Core6ContractTests(unittest.TestCase):
    def test_exact_90_unique_runs(self):
        rows=plan_rows()
        self.assertEqual(len(rows),90)
        self.assertEqual(len({(r['target'],r['recipe'],r['seed']) for r in rows}),90)
        self.assertEqual([r['index'] for r in rows],list(range(90)))
        self.assertEqual(collections.Counter(r['target'] for r in rows),dict.fromkeys(TARGETS,30))
        for r in rows:
            self.assertEqual((r['epochs'],r['precision'],r['batch'],r['eval_batch']),(100,'fp16',16,8))
    def test_twelve_smokes_cover_every_recipe_and_target(self):
        rows=smoke_rows()
        self.assertEqual(len(rows),12)
        self.assertEqual({r['recipe'] for r in rows},set(RECIPES))
        self.assertEqual({r['target'] for r in rows},set(TARGETS))
    def test_actual_recorded_split_counts_and_label_map(self):
        rows=[]
        for freq in TARGETS:
            rr=read_csv(KIT/f'manifests/core6_source/{freq}.csv')
            self.assertEqual(dict(collections.Counter(r['split'] for r in rr)),SPLIT_COUNTS[freq])
            for r in rr:
                self.assertEqual(int(r['class_id']),CLASSES6.index(r['class_name']))
            rows+=rr
        self.assertEqual(len(rows),1336)
        for target in TARGETS:
            n={split:sum(r['split']==split and ((r['frequency']==target) if split=='test' else (r['frequency']!=target)) for r in rows) for split in ('train','val','test')}
            self.assertEqual(n,EXPECTED_N[target])
    def test_all_target_recipe_configs_six_class_and_no_target_training(self):
        for target in TARGETS:
            for recipe in RECIPES:
                c=make_config(recipe,'core6',target,100,16,'fp16')
                self.assertEqual(c['NUM_CLASSES'],6)
                self.assertEqual(c['CLASSES'],CLASSES6)
                self.assertNotIn(target,c['TRAIN_FREQS'])
                self.assertEqual(c['TEST_FREQS'],[target])
                self.assertEqual(c['FREQ_TO_IDX'],{f:i for i,f in enumerate(c['TRAIN_FREQS'])})
                self.assertFalse(c['USE_DANN'])
                self.assertEqual(c['V13_DECORR_WEIGHT'],0)
                for k in ('V15_FALSIFY_WEIGHT','V15_KIN_SOURCE_WEIGHT','V15_SENSOR_UNIFORM_WEIGHT','V15_CONSIST_WEIGHT','V15R_MARGIN_WEIGHT'):
                    self.assertEqual(c[k],0)
    def test_core_ablation_differences(self):
        c=lambda r:make_config(r,'core6','77GHz',100,16,'fp16')
        full=c('proposed'); residual=c('residual_only'); grl=c('grl_only'); disc=c('discrete_acr')
        self.assertEqual((full['V13_FREQ_WEIGHT'],full['V13_GRL_WEIGHT']),(0.05,0.3))
        self.assertEqual(residual['V13_GRL_WEIGHT'],0)
        self.assertEqual(grl['V13_FREQ_WEIGHT'],0)
        self.assertTrue(disc['V13_GRL_DISCRETE']);self.assertEqual(disc['V13_GRL_BINS'],3)
        erm=c('no_image_aug');stretch=c('stretch_only')
        self.assertFalse(any(erm[k] for k in ('USE_DAS','USE_HFT','USE_SPEC_AUGMENT')))
        self.assertTrue(stretch['USE_DAS']);self.assertFalse(stretch['USE_HFT']);self.assertFalse(stretch['USE_SPEC_AUGMENT'])
        self.assertEqual((c('wide_acr')['DAS_FIXED_FULL_F_LOW_GHZ'],c('wide_acr')['DAS_FIXED_FULL_F_HIGH_GHZ']),(15.,140.))
        self.assertEqual((c('narrow_acr')['DAS_FIXED_NARROW_F_LOW_GHZ'],c('narrow_acr')['DAS_FIXED_NARROW_F_HIGH_GHZ']),(10.,30.))
    def test_original_source_integrity(self):
        self.assertTrue(verify_source(KIT/'evidence'))
    def test_controller_declares_cuda_guard_scaler_and_fixed_final(self):
        source=(KIT/'train_controlled.py').read_text()
        ast.parse(source)
        self.assertIn("torch.amp.GradScaler('cuda'",source)
        self.assertIn('scaler.unscale_(opt)',source)
        self.assertIn('scaler.step(opt)',source)
        self.assertIn('scaler.update()',source)
        self.assertIn('if not torch.cuda.is_available()',source)
        self.assertLess(source.index("write_json(a.out/'selection_frozen_before_target.json'"),source.index("load_rows(folder,'test',protocol)"))
        self.assertNotIn('pool_ep100_ema.pt',source)

if __name__=='__main__':unittest.main(verbosity=2)
