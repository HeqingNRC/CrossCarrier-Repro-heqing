#!/usr/bin/env python3
"""Synthetic artifact tests: verify report identity checks and missing-run handling.
These fixtures do not contain trained models or research results.
"""
import contextlib,csv,io,json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from suite_constants import plan_rows,TARGETS,EXPECTED_N
from xccontrol.common import CLASSES6,write_json,sha256
from xccontrol.recipes import make_config
from xccontrol.metrics import classification
import summarize_suite as S

class SummaryFixtureTest(unittest.TestCase):
    def test_complete_matrix_and_rejection_of_wrong_job(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);batch=root/'batches/SYNTHETIC_NOT_A_REAL_RUN';batch.mkdir(parents=True)
            (batch/'runs').mkdir()
            old_kit=S.KIT;S.KIT=root
            try:
                submission=dict(train_job_id='999999999')
                plan=dict(tasks=plan_rows())
                write_json(batch/'submission.json',submission);write_json(batch/'plan.json',plan)
                protos={};labels={};paths={}
                for target in TARGETS:
                    folder=root/'prepared'/f'core6_to_{target}';folder.mkdir(parents=True)
                    n=EXPECTED_N[target]['test'];labels[target]=np.arange(n)%6
                    paths[target]=[f'/SYNTHETIC_NOT_REAL/{target}/{i}.png' for i in range(n)]
                    with (folder/'test.csv').open('w',newline='') as f:
                        w=csv.DictWriter(f,fieldnames=['path','class_idx_7c']);w.writeheader()
                        w.writerows(dict(path=p,class_idx_7c=int(y)) for p,y in zip(paths[target],labels[target]))
                    protos[target]=dict(source_hashes={'dummy_source':'SYNTHETIC'},classes=CLASSES6,target=target)
                    write_json(folder/'protocol.json',protos[target])
                for row in plan['tasks']:
                    target=row['target'];out=batch/'runs'/f'{row["index"]:03d}_{target}_{row["recipe"]}_seed{row["seed"]}'
                    (out/'checkpoints').mkdir(parents=True);(out/'predictions').mkdir()
                    checkpoint=out/'checkpoints/final_ema.pt';checkpoint.write_bytes(b'SYNTHETIC HASH FIXTURE, NOT A MODEL')
                    L=np.eye(6,dtype=np.float32)[labels[target]]
                    pp=out/'predictions/final_ema.npz'
                    np.savez_compressed(pp,paths=np.array(paths[target]),labels=labels[target],classes=np.array(CLASSES6),logits=L)
                    m=classification(labels[target],L.argmax(1),CLASSES6)
                    write_json(out/'predictions/final_ema.json',m)
                    args={k:row[k] for k in ['dataset','target','recipe','seed','epochs','batch','eval_batch','precision']};args['smoke_steps']=0
                    meta=dict(arguments=args,protocol=protos[target],packages={'test':'SYNTHETIC'},gpu='NO_REAL_GPU',
                              array_job_id='999999999',array_task_id=str(row['index']),batch_id=batch.name)
                    write_json(out/'run_manifest.json',meta)
                    write_json(out/'effective_config.json',make_config(row['recipe'],'core6',target,100,16,'fp16'))
                    write_json(out/'history.json',[dict(epoch=e,target_evaluated=False) for e in range(1,101)])
                    frozen=dict(epoch=100,selection='fixed_epoch100_ema',target_used_for_training_or_selection=False,
                                checkpoint_sha256=sha256(checkpoint),amp_skipped_steps=0)
                    write_json(out/'selection_frozen_before_target.json',frozen)
                    write_json(out/'DONE.json',dict(status='COMPLETE',epoch=100,array_job_id='999999999',
                        array_task_id=str(row['index']),batch_id=batch.name,checkpoint_sha256=sha256(checkpoint),
                        predictions_sha256=sha256(pp),results={'final_ema':m}))
                with contextlib.redirect_stdout(io.StringIO()):report=S.summarize(batch)
                self.assertEqual(report['status'],'COMPLETE_90_RUNS')
                self.assertEqual(len(report['rows']),30)
                self.assertEqual(len(report['paired_deltas']),33)
                self.assertTrue((batch/'COMPLETE90.json').exists())
                first=next((batch/'runs').glob('000_*'))/'run_manifest.json'
                m=json.loads(first.read_text());m['array_job_id']='123_OLD_JOB';write_json(first,m)
                with contextlib.redirect_stdout(io.StringIO()):report=S.summarize(batch)
                self.assertEqual(report['status'],'INCOMPLETE_OR_INVALID')
                self.assertEqual(report['completed_validated_runs'],89)
                self.assertEqual(len(report['rows']),29)
                self.assertFalse((batch/'COMPLETE90.json').exists())
                self.assertTrue(any('wrong batch/job identity' in x['error'] for x in report['issues']))
            finally:S.KIT=old_kit

if __name__=='__main__':unittest.main(verbosity=2)
