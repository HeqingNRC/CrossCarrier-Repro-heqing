#!/usr/bin/env python3
"""Core6-only prepare -> GPU smoke -> 90 fresh train/eval runs -> checked summary.
No old job or old task checkpoint is used by this workflow.
"""
from __future__ import annotations
import argparse, datetime as dt, json, os, re, subprocess, sys, uuid
from pathlib import Path
from suite_constants import KIT, DEFAULT_ASSETS, DEFAULT_PYTHON, plan_rows, smoke_rows
from xccontrol.common import sha256, write_json, verify_source


def verify_delivery():
    path=KIT/'DELIVERY_HASHES.json'
    expected=json.loads(path.read_text())
    bad=[]
    for rel,digest in expected.items():
        p=KIT/rel
        if not p.is_file() or sha256(p)!=digest:
            bad.append(rel)
    if bad:
        raise RuntimeError('Package changed after release: '+', '.join(bad))
    verify_source(KIT/'evidence')
    return sha256(path)


def check_runtime():
    verify_delivery()
    runtime=json.loads((KIT/'runtime.json').read_text())
    if runtime['kit']!=str(KIT):
        raise RuntimeError('Prepared toolkit was moved; do not submit against stale absolute paths')
    if sha256(KIT/'DELIVERY_HASHES.json')!=runtime['delivery_sha256']:
        raise RuntimeError('Prepared data belongs to a different code release')
    return runtime


def prepare(a):
    digest=verify_delivery()
    if (KIT/'runtime.json').exists():
        raise FileExistsError('runtime.json already exists: prepared suite will not be overwritten. Use submit/report.')
    if sys.version_info[:2] != (3,10):
        raise RuntimeError('Use the existing xcarrier_20260922 Python 3.10 environment')
    import importlib.metadata as md
    versions={p:md.version(p) for p in ['torch','torchvision','timm','peft','transformers','numpy','pandas','Pillow','safetensors']}
    if not versions['torch'].startswith('2.5.1') or not versions['torchvision'].startswith('0.20.1'):
        raise RuntimeError(f'Expected the verified Torch 2.5.1 / torchvision 0.20.1 environment: {versions}')
    assets=a.assets.expanduser().resolve()
    weights=assets/'weights'
    backbone=weights/'hub/models--timm--vit_large_patch16_dinov3.lvd1689m'
    candidates=list((backbone/'snapshots').glob('*/model.safetensors'))
    if not candidates or not any(p.is_file() and p.stat().st_size>100_000_000 for p in candidates):
        raise FileNotFoundError(f'Bundled DINOv3 weights missing: {backbone}')
    link=KIT/'evidence/weights'
    if link.is_symlink():
        if link.resolve()!=weights:
            raise RuntimeError(f'Weights link points to a different cache: {link}')
    elif link.exists():
        raise FileExistsError(f'Refusing to overwrite {link}')
    else:
        link.symlink_to(weights,target_is_directory=True)
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
               PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',TERM='dumb')
    print('Checking the installed environment and CPU-only objective tests...',flush=True)
    subprocess.run([sys.executable,'-m','pip','check'],check=True,env=env)
    subprocess.run([sys.executable,str(KIT/'tests/test_core6_suite.py')],cwd=KIT,check=True,env=env)
    subprocess.run([sys.executable,str(KIT/'tests/self_test.py'),'--repo',str(KIT/'evidence'),
                    '--out',str(KIT/'CPU_TEST_SOCKEYE.json')],cwd=KIT,check=True,env=env)
    from prepare_core6 import prepare_core6
    summary=prepare_core6(KIT/'evidence',KIT/'prepared',
                         a.core_root.expanduser().resolve() if a.core_root else None,
                         a.preserve_legacy_splits)
    runtime=dict(kit=str(KIT),python=str(Path(sys.executable).absolute()),assets=str(assets),
                 source=str(KIT/'evidence'),prepared=str(KIT/'prepared'),
                 delivery_sha256=digest,versions=versions,
                 preserve_legacy_splits=a.preserve_legacy_splits,
                 protocol_hashes={t['target']:sha256(KIT/'prepared'/f'core6_to_{t["target"]}'/'protocol.json')
                                  for t in summary['protocols']},
                 status='DATA_AND_CPU_CHECKS_PASSED_NO_GPU_JOB_SUBMITTED')
    write_json(KIT/'runtime.json',runtime)
    print('\nPREPARE_OK: Core6 six classes; three recorded folds; 90 formal experiments planned; no GPU job submitted.')


def sbatch(command):
    p=subprocess.run(command,text=True,capture_output=True)
    if p.returncode:
        raise RuntimeError(f'sbatch failed ({p.returncode}): {p.stdout}\n{p.stderr}')
    raw=p.stdout.strip()
    jid=raw.split(';',1)[0]
    if not re.fullmatch(r'\d+',jid):
        raise RuntimeError('Unexpected sbatch --parsable response: '+raw)
    if p.stderr.strip():
        print(p.stderr, file=sys.stderr)
    return jid,raw


def submit(a):
    runtime=check_runtime()
    if (KIT/'CURRENT_BATCH.txt').exists() and not a.new_batch:
        raise RuntimeError('A batch already exists. Use report; --new-batch is required to submit another 90 runs.')
    if not 1<=a.max_concurrent<=12:
        raise ValueError('Use a concurrency between 1 and 12')
    stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    batch=KIT/'batches'/('batch-'+stamp+'-'+uuid.uuid4().hex[:6])
    batch.mkdir(parents=True,exist_ok=False)
    for name in ('logs','runs','smoke','status'):
        (batch/name).mkdir()
    rows=plan_rows()
    smoke=smoke_rows()
    write_json(batch/'plan.json',dict(dataset='core6',tasks=rows,smoke_tasks=smoke,
        runtime=runtime,complete_requires=90,selection='fixed_epoch100_ema',
        created_utc=stamp,baseline_source='new proposed runs in this batch only'))
    ledger=['index,target,recipe,alias,seed,epochs,batch,precision']
    ledger += [f'{r["index"]},{r["target"]},{r["recipe"]},{r["alias"]},{r["seed"]},100,16,fp16' for r in rows]
    (batch/'experiment_ledger.csv').write_text('\n'.join(ledger)+'\n')
    record=dict(batch=str(batch),status='SUBMITTING',smoke_job_id=None,train_job_id=None,
                expected_train_runs=90,expected_smoke_runs=len(smoke),
                max_concurrent=a.max_concurrent)
    write_json(batch/'submission.json',record)
    (KIT/'CURRENT_BATCH.txt').write_text(str(batch)+'\n')
    common=['sbatch','--parsable','--account=st-zliu-1-gpu','--partition=gpu','--nodes=1','--ntasks=1',
            '--cpus-per-task=4','--mem=32G','--gres=gpu:1',f'--chdir={KIT}']
    try:
        jid,raw=sbatch(common+['--job-name=core6-smoke',f'--array=0-{len(smoke)-1}%1',
            '--time=00:45:00',f'--output={batch}/logs/smoke-%A_%a.out',f'--error={batch}/logs/smoke-%A_%a.err',
            str(KIT/'run_gpu.sbatch'),str(KIT),'smoke',str(batch),runtime['python']])
        record.update(smoke_job_id=jid,smoke_sbatch_response=raw)
        write_json(batch/'submission.json',record)
        print('NEW CORE6 SMOKE ARRAY = '+jid,flush=True)
        # The formal array cannot start unless every smoke task succeeds.
        train,raw=sbatch(common+['--job-name=core6-ablation',f'--array=0-89%{a.max_concurrent}',
            '--time=06:00:00',f'--dependency=afterok:{jid}',
            f'--output={batch}/logs/train-%A_%a.out',f'--error={batch}/logs/train-%A_%a.err',
            str(KIT/'run_gpu.sbatch'),str(KIT),'run',str(batch),runtime['python']])
        record.update(train_job_id=train,train_sbatch_response=raw,status='SUBMITTED')
        write_json(batch/'submission.json',record)
        (KIT/'CURRENT_BATCH.txt').write_text(str(batch)+'\n')
        print('NEW CORE6 TRAIN90 ARRAY = '+train,flush=True)
        print('BATCH = '+str(batch),flush=True)
        print('Training depends on the new smoke array. No historical job is a dependency.',flush=True)
        print('The last completed worker attempts the checked CPU-only summary; report also rebuilds it.')
    except Exception as e:
        record.update(status='SUBMISSION_PARTIAL_OR_FAILED',error=str(e))
        write_json(batch/'submission.json',record)
        print('Submission record retained at '+str(batch/'submission.json'),file=sys.stderr)
        raise


def read_batch(path):
    path=path.resolve()
    plan=json.loads((path/'plan.json').read_text())
    if plan['tasks'] != plan_rows() or plan['smoke_tasks'] != smoke_rows():
        raise RuntimeError('Experiment matrix was modified after submission')
    runtime=check_runtime()
    if plan['runtime']!=runtime:
        raise RuntimeError('Runtime changed after submission')
    for target,digest in runtime['protocol_hashes'].items():
        if sha256(KIT/'prepared'/f'core6_to_{target}'/'protocol.json')!=digest:
            raise RuntimeError('Protocol metadata changed after submission')
    return plan


def worker(a):
    batch=a.batch.resolve()
    plan=read_batch(batch)
    index=int(os.environ['SLURM_ARRAY_TASK_ID'])
    smoking=a.command=='smoke'
    rows=plan['smoke_tasks'] if smoking else plan['tasks']
    row=rows[index]
    out=batch/('smoke' if smoking else 'runs')/f'{index:03d}_{row["target"]}_{row["recipe"]}_seed{row["seed"]}'
    # Validate dependency completion at the application level as well.
    if not smoking:
        for r in plan['smoke_tasks']:
            p=batch/'smoke'/f'{r["index"]:03d}_{r["target"]}_{r["recipe"]}_seed{r["seed"]}'/'SMOKE_PASS.json'
            if not p.is_file() or json.loads(p.read_text()).get('status')!='PASS':
                raise RuntimeError(f'Missing successful new smoke: {p}')
    runtime=plan['runtime']
    cmd=[runtime['python'],str(KIT/'train_controlled.py'),'--repo',runtime['source'],
         '--prepared',runtime['prepared'],'--dataset','core6','--target',row['target'],
         '--recipe',row['recipe'],'--seed',str(row['seed']),'--out',str(out),
         '--epochs','100','--batch','16','--eval-batch','8','--precision','fp16']
    if smoking:
        cmd+=['--smoke-steps','2']
    if runtime['preserve_legacy_splits']:
        cmd+=['--acknowledge-duplicate-risk']
    env=os.environ.copy(); env['CORE6_BATCH_ID']=batch.name
    print('COMMAND: '+' '.join(cmd),flush=True)
    status_path=batch/'status'/f'{"smoke" if smoking else "train"}-{index:03d}.json'
    write_json(status_path,dict(status='RUNNING',out=str(out),task=row,job_id=os.environ.get('SLURM_JOB_ID')))
    p=subprocess.run(cmd,cwd=KIT,env=env)
    if p.returncode:
        write_json(status_path,dict(status='FAILED',returncode=p.returncode,out=str(out),task=row,
            job_id=os.environ.get('SLURM_JOB_ID')))
        raise SystemExit(p.returncode if p.returncode>0 else 1)
    expected='SMOKE_PASS.json' if smoking else 'DONE.json'
    if not (out/expected).is_file():
        raise RuntimeError('Child exited without its completion record: '+str(out/expected))
    write_json(status_path,dict(status='COMPLETE',out=str(out),task=row,job_id=os.environ.get('SLURM_JOB_ID')))
    if not smoking:
        # Small metadata check only. The last worker performs one full validation.
        done=list((batch/'status').glob('train-*.json'))
        if len(done)==90 and all(json.loads(p.read_text()).get('status')=='COMPLETE' for p in done):
            import fcntl
            with (batch/'.summary.lock').open('w') as f:
                try:
                    fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    return
                try:
                    from summarize_suite import summarize
                    summarize(batch)
                except Exception as e:
                    write_json(batch/'SUMMARY_ERROR.json',dict(error=str(e),training_outputs_preserved=True))
                    print('Training completed; summary validation failed: '+str(e),file=sys.stderr)


def report(a):
    batch=a.batch
    if batch is None:
        batch=Path((KIT/'CURRENT_BATCH.txt').read_text().strip())
    read_batch(batch)
    rec=json.loads((batch/'submission.json').read_text())
    print(json.dumps(rec,indent=2))
    ids=[rec.get('smoke_job_id'),rec.get('train_job_id')]
    for jid in ids:
        if jid:
            subprocess.run(['sacct','-j',jid,'--format=JobID%22,JobName%20,State,Elapsed,ExitCode'])
    from summarize_suite import summarize
    import fcntl
    with (batch/'.summary.lock').open('w') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        summarize(batch)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare')
    p.add_argument('--assets',type=Path,default=DEFAULT_ASSETS)
    p.add_argument('--core-root',type=Path)
    p.add_argument('--preserve-legacy-splits',action='store_true')
    p=sub.add_parser('submit')
    p.add_argument('--max-concurrent',type=int,default=3)
    p.add_argument('--new-batch',action='store_true',help='Explicitly request a separate repeat of all 90 runs')
    for name in ('smoke','run'):
        p=sub.add_parser(name);p.add_argument('--batch',type=Path,required=True)
    p=sub.add_parser('report');p.add_argument('--batch',type=Path)
    a=ap.parse_args()
    if a.command=='prepare':prepare(a)
    elif a.command=='submit':submit(a)
    elif a.command in ('smoke','run'):worker(a)
    else:report(a)

if __name__=='__main__':
    main()
