#!/usr/bin/env python3
"""CPU-only report restricted to the explicit 90-run batch manifest.
Incomplete/mismatched runs are named and never filled with historical values.
"""
from __future__ import annotations
import csv,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np
from suite_constants import KIT, RECIPES, TARGETS, SEEDS, EXPECTED_N, ALIASES, plan_rows
from xccontrol.common import CLASSES6, sha256, write_json, read_csv
from xccontrol.recipes import make_config
from xccontrol.metrics import classification
from aggregate import ensemble_predictions


def inspect_run(batch, row, submission, plan):
    out=batch/'runs'/f'{row["index"]:03d}_{row["target"]}_{row["recipe"]}_seed{row["seed"]}'
    if not (out/'DONE.json').is_file():
        status=batch/'status'/f'train-{row["index"]:03d}.json'
        status=json.loads(status.read_text()).get('status','INCOMPLETE') if status.is_file() else 'NOT_STARTED_OR_NO_RECORD'
        raise RuntimeError(f'{status}: {out.name}')
    if (out/'SMOKE_PASS.json').exists():
        raise RuntimeError('A smoke result was placed in a formal run folder')
    done=json.loads((out/'DONE.json').read_text())
    meta=json.loads((out/'run_manifest.json').read_text())
    config=json.loads((out/'effective_config.json').read_text())
    frozen=json.loads((out/'selection_frozen_before_target.json').read_text())
    hist=json.loads((out/'history.json').read_text())
    args=meta['arguments']
    for k in ['dataset','target','recipe','seed','epochs','batch','eval_batch','precision']:
        if args[k] != row[k]:
            raise RuntimeError(f'{out.name}: mismatched {k}={args.get(k)!r}')
    for k in ('array_job_id','array_task_id','batch_id'):
        expected={'array_job_id':submission['train_job_id'],'array_task_id':str(row['index']),'batch_id':batch.name}[k]
        if str(meta.get(k))!=str(expected) or str(done.get(k))!=str(expected):
            raise RuntimeError(f'{out.name}: wrong batch/job identity {k}')
    if args.get('smoke_steps')!=0 or done.get('status')!='COMPLETE' or done.get('epoch')!=100:
        raise RuntimeError('Formal completion markers invalid')
    if len(hist)!=100 or [r.get('epoch') for r in hist]!=list(range(1,101)):
        raise RuntimeError('Epoch history is not exactly 1..100')
    if any(r.get('target_evaluated') is not False for r in hist):
        raise RuntimeError('Target used before final epoch')
    if frozen.get('epoch')!=100 or frozen.get('selection')!='fixed_epoch100_ema' or frozen.get('target_used_for_training_or_selection') is not False:
        raise RuntimeError('Final-EMA selection receipt is invalid')
    expected_cfg=make_config(row['recipe'],'core6',row['target'],100,16,'fp16')
    for k,v in expected_cfg.items():
        if config.get(k)!=v:
            raise RuntimeError(f'{out.name}: effective recipe differs at {k}')
    proto_path=KIT/'prepared'/f'core6_to_{row["target"]}'
    proto=json.loads((proto_path/'protocol.json').read_text())
    if proto!=meta['protocol']:
        raise RuntimeError('Run used different protocol metadata')
    checkpoint=out/'checkpoints/final_ema.pt'
    digest=sha256(checkpoint)
    if digest != done['checkpoint_sha256'] or digest != frozen['checkpoint_sha256']:
        raise RuntimeError('Final checkpoint differs from its completion receipt')
    pred_path=out/'predictions/final_ema.npz'
    if sha256(pred_path)!=done['predictions_sha256']:
        raise RuntimeError('Prediction archive changed after evaluation')
    with np.load(pred_path,allow_pickle=False) as z:
        data={k:z[k].copy() for k in z.files}
    rows=read_csv(proto_path/'test.csv')
    expected_labels=np.array([int(r['class_idx_7c']) for r in rows])
    if not np.array_equal(data['paths'],[r['path'] for r in rows]) or not np.array_equal(data['labels'],expected_labels):
        raise RuntimeError('Target image order/labels differ from fixed test split')
    if not np.array_equal(data['classes'],CLASSES6) or data['logits'].shape!=(EXPECTED_N[row['target']]['test'],6):
        raise RuntimeError('Not the prescribed Core6 six-class test set')
    if not np.isfinite(data['logits']).all():
        raise RuntimeError('Non-finite saved logits')
    metrics=classification(data['labels'],data['logits'].argmax(1),CLASSES6)
    stored=json.loads((out/'predictions/final_ema.json').read_text())
    for name in ['accuracy','macro_f1']:
        if abs(metrics[name]-stored[name])>1e-12 or abs(metrics[name]-done['results']['final_ema'][name])>1e-12:
            raise RuntimeError('Saved metrics differ from recomputed predictions')
    return dict(run=str(out),row=row,metrics=metrics,data=data,packages=meta['packages'],
                gpu=meta['gpu'],amp_skips=frozen['amp_skipped_steps'],
                source_hashes=meta['protocol']['source_hashes'])


def summarize(batch):
    batch=Path(batch).resolve()
    submission=json.loads((batch/'submission.json').read_text())
    plan=json.loads((batch/'plan.json').read_text())
    if plan['tasks']!=plan_rows():
        raise RuntimeError('Unexpected experiment plan')
    valid=[];issues=[]
    for row in plan['tasks']:
        try:
            valid.append(inspect_run(batch,row,submission,plan))
        except Exception as e:
            issues.append(dict(index=row['index'],target=row['target'],recipe=row['recipe'],seed=row['seed'],error=str(e)))
    # Do not aggregate across changed software/source snapshots.
    if valid:
        first=valid[0]
        for v in valid:
            if v['packages']!=first['packages'] or v['source_hashes']!=first['source_hashes']:
                raise RuntimeError('Mixed dependency versions or source files within this batch')
    groups=defaultdict(list)
    for v in valid:
        groups[(v['row']['target'],v['row']['recipe'])].append(v)
    summary_rows=[];ensemble_rows=[];per_seed_rows=[];confusion_rows=[]
    f1_by_group={}
    for t in TARGETS:
        for recipe in RECIPES:
            vv=groups.get((t,recipe),[])
            if {v['row']['seed'] for v in vv}!=set(SEEDS) or len(vv)!=3:
                continue
            vv=sorted(vv,key=lambda v:SEEDS.index(v['row']['seed']))
            data=vv[0]['data']
            for v in vv[1:]:
                for key in ['paths','classes','labels']:
                    if not np.array_equal(data[key],v['data'][key]):
                        raise RuntimeError('Cannot ensemble mismatched images/classes')
            L=np.stack([v['data']['logits'] for v in vv])
            f1=np.array([v['metrics']['macro_f1'] for v in vv])
            acc=np.array([v['metrics']['accuracy'] for v in vv])
            ens={k:classification(data['labels'],p,CLASSES6) for k,p in ensemble_predictions(L).items()}
            f1_by_group[(t,recipe)]=f1
            summary_rows.append(dict(target=t,recipe=recipe,original_name=ALIASES[recipe],n=len(data['labels']),
                seeds=SEEDS,macro_f1_mean=float(f1.mean()),macro_f1_std_population=float(f1.std()),
                accuracy_mean=float(acc.mean()),accuracy_std_population=float(acc.std()),
                macro_f1_per_seed=f1.tolist(),accuracy_per_seed=acc.tolist(),
                logit_ensemble_macro_f1=ens['logit_average']['macro_f1'],
                logit_ensemble_accuracy=ens['logit_average']['accuracy']))
            for k,m in ens.items():
                ensemble_rows.append(dict(target=t,recipe=recipe,rule=k,metrics=m))
            for v in vv:
                per_seed_rows.append(dict(target=t,recipe=recipe,seed=v['row']['seed'],
                    accuracy=v['metrics']['accuracy'],macro_f1=v['metrics']['macro_f1'],
                    gpu=v['gpu'],amp_skipped_steps=v['amp_skips'],run=v['run']))
                for ci,c in enumerate(CLASSES6):
                    for pi,p in enumerate(CLASSES6):
                        confusion_rows.append(dict(target=t,recipe=recipe,seed=v['row']['seed'],
                             true_class=c,predicted_class=p,count=v['metrics']['confusion'][ci][pi]))
    # Paired seed comparisons, not significance tests across three seeds.
    comparisons=[('stretch_only','no_image_aug','Stretch-only minus no-image-augmentation'),
                 ('radar_aug','no_image_aug','Generic radar augmentation minus no-image-augmentation'),
                 ('das_only','radar_aug','Adding stretch to generic radar augmentation'),
                 ('das_only','stretch_only','Adding generic radar augmentation to stretch'),
                 ('proposed','radar_aug','Joint DAS/ACR package minus matched radar-augmentation control'),
                 ('proposed','das_only','Adding full ACR to full DAS'),
                 ('proposed','residual_only','Adding continuous GRL with residual retained'),
                 ('proposed','grl_only','Adding residual with continuous GRL retained'),
                 ('proposed','discrete_acr','Continuous minus three-bin carrier adversary'),
                 ('proposed','wide_acr','Curriculum minus fixed-wide range (range and schedule both differ)'),
                 ('proposed','narrow_acr','Curriculum minus fixed-narrow range (range and schedule both differ)')]
    deltas=[]
    for t in TARGETS:
        for treatment,control,label in comparisons:
            if (t,treatment) in f1_by_group and (t,control) in f1_by_group:
                d=100*(f1_by_group[(t,treatment)]-f1_by_group[(t,control)])
                deltas.append(dict(target=t,treatment=treatment,control=control,comparison=label,
                                  mean_delta_pp=float(d.mean()),std_population_pp=float(d.std()),
                                  paired_seed_deltas_pp=d.tolist(),seeds=SEEDS))
    report_dir=batch/'summary';report_dir.mkdir(exist_ok=True)
    report=dict(status='COMPLETE_90_RUNS' if len(valid)==90 and not issues else 'INCOMPLETE_OR_INVALID',
                batch=str(batch),train_array_id=submission.get('train_job_id'),
                completed_validated_runs=len(valid),expected_runs=90,
                complete_groups=len(summary_rows),expected_groups=30,issues=issues,
                rows=summary_rows,ensembles=ensemble_rows,paired_deltas=deltas,
                selector='fixed_epoch100_ema',std='population across prescribed seeds',
                notes=['No historical score, run or checkpoint fills a missing row.',
                       'Seed-paired deltas are descriptive; no significance claim is automatic.',
                       'Six-class historical image splits retain documented duplicate risks.',
                       'Fixed-range comparisons change both carrier range and its schedule.',
                       'No-image-augmentation retains SupCon and feature anchoring; it is not CE-only ERM.'])
    write_json(report_dir/'summary.json',report)
    def write_table(name,rows):
        if not rows:
            (report_dir/name).write_text('',encoding='utf-8')
            return
        with (report_dir/name).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_table('summary.csv',summary_rows);write_table('per_seed.csv',per_seed_rows)
    write_table('paired_deltas.csv',deltas);write_table('confusion_matrices.csv',confusion_rows)
    lines=['# CoreMovements6: fresh 90-run reproduction and ablation',
           '',f'Status: **{report["status"]}**; validated {len(valid)}/90 runs, {len(summary_rows)}/30 complete three-seed groups.',
           f'Batch: `{batch.name}`; training array: `{submission.get("train_job_id")}`.',
           'Six classes; targets 10/24/77 GHz; seeds 42/1234/31415; 100 epochs; FP16; fixed final EMA.',
           '', '| Target | Recipe | Test N | Accuracy mean +/- SD | Macro-F1 mean +/- SD | Logit ensemble F1 |',
           '|---|---|---:|---:|---:|---:|']
    for r in summary_rows:
        lines.append(f'| {r["target"]} | {r["recipe"]} | {r["n"]} | {r["accuracy_mean"]:.4f} +/- {r["accuracy_std_population"]:.4f} | {r["macro_f1_mean"]:.4f} +/- {r["macro_f1_std_population"]:.4f} | {r["logit_ensemble_macro_f1"]:.4f} |')
    lines+=['','## Seed-paired component comparisons (percentage points)',
            '| Target | Treatment - control | Mean delta | Per-seed deltas [42,1234,31415] |',
            '|---|---|---:|---|']
    for d in deltas:
        lines.append(f'| {d["target"]} | {d["treatment"]} - {d["control"]} | {d["mean_delta_pp"]:+.2f} | '+', '.join(f'{x:+.2f}' for x in d['paired_seed_deltas_pp'])+' |')
    if issues:
        lines+=['','## Missing or invalid tasks (no historical fallback)']
        lines += [f'- Task {r["index"]}, {r["target"]}/{r["recipe"]}/seed{r["seed"]}: {r["error"]}' for r in issues]
    lines+=['','## Interpretation limits'] + ['- '+x for x in report['notes']]
    text='\n'.join(lines)+'\n'
    tmp=report_dir/'summary.md.tmp';tmp.write_text(text,encoding='utf-8');tmp.replace(report_dir/'summary.md')
    if report['status']=='COMPLETE_90_RUNS':
        write_json(batch/'COMPLETE90.json',dict(status='COMPLETE_90_RUNS',summary_sha256=sha256(report_dir/'summary.json'),
                                              train_array_id=submission['train_job_id']))
    elif (batch/'COMPLETE90.json').exists():
        (batch/'COMPLETE90.json').unlink()
    print(text)
    return report
