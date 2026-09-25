#!/usr/bin/env python3
"""Generate a bounded Slurm array and experiment ledger. DOES NOT SUBMIT IT."""
from __future__ import annotations
import argparse,csv,json,shlex
from pathlib import Path
from xccontrol.common import KIT,write_json,FREQUENCIES
from xccontrol.recipes import RECIPES,EXPECTED_F1_ORIGINAL7


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset',choices=['original7','core6'],required=True)
    ap.add_argument('--target',choices=FREQUENCIES+['all'],default='77GHz')
    ap.add_argument('--recipes',nargs='+',choices=list(RECIPES)+['all'],default=['proposed'])
    ap.add_argument('--seeds',type=int,nargs='+',default=[42,1234,31415])
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--epochs',type=int,default=100)
    ap.add_argument('--precision',choices=['bf16','fp32'],default='bf16')
    ap.add_argument('--max-concurrent',type=int,default=1)
    ap.add_argument('--time',default='12:00:00')
    ap.add_argument('--swad',action='store_true')
    ap.add_argument('--fishr',action='store_true')
    ap.add_argument('--source-only',action='store_true')
    ap.add_argument('--fishr-lambda',type=float,default=None)
    ap.add_argument('--fishr-anneal',type=int,default=1500)
    ap.add_argument('--acknowledge-duplicate-risk',action='store_true')
    a=ap.parse_args()
    if a.fishr and a.fishr_lambda is None:raise ValueError('Specify a source-validation pilot candidate with --fishr-lambda; no implicit reuse of 1000.')
    if a.out.exists():raise FileExistsError(a.out)
    if a.out.resolve().parent!=KIT/'plans':raise ValueError('Use --out plans/NAME inside the toolkit.')
    if a.max_concurrent<1 or not a.seeds or len(set(a.seeds))!=len(a.seeds):raise ValueError('Invalid concurrency or seeds')
    targets=FREQUENCIES if a.target=='all' else [a.target]
    if a.dataset=='original7' and targets!=['77GHz']:raise ValueError('Original study has only target77')
    recipes=list(RECIPES) if 'all' in a.recipes else a.recipes
    rows=[]
    for target in targets:
        for recipe in recipes:
            for seed in a.seeds:
                name=f'{a.dataset}_{target}_{recipe}_seed{seed}'
                cmd=['train_controlled.py','--dataset',a.dataset,'--target',target,'--recipe',recipe,
                     '--seed',str(seed),'--epochs',str(a.epochs),'--precision',a.precision]
                if a.swad:cmd+=['--swad']
                if a.source_only:cmd+=['--source-only']
                if a.fishr:cmd+=['--fishr','--fishr-lambda',str(a.fishr_lambda),'--fishr-anneal',str(a.fishr_anneal)]
                if a.acknowledge_duplicate_risk:cmd+=['--acknowledge-duplicate-risk']
                rows.append(dict(index=len(rows),name=name,dataset=a.dataset,target=target,recipe=recipe,seed=seed,
                                 expected_original7_f1=EXPECTED_F1_ORIGINAL7[recipe] if a.dataset=='original7' else None,
                                 status='NOT_RUN',command=cmd))
    a.out.mkdir(parents=True)
    write_json(a.out/'jobs.json',rows)
    with (a.out/'ledger.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['index','name','dataset','target','recipe','seed','expected_original7_f1','status'])
        w.writeheader();w.writerows({k:r[k] for k in w.fieldnames} for r in rows)
    planrel=str(a.out.resolve().relative_to(KIT))
    (a.out/'run_one.py').write_text('''from pathlib import Path
import json,os,subprocess,sys
root=Path(__file__).resolve().parents[2]
rows=json.loads((Path(__file__).parent/'jobs.json').read_text())
i=int(os.environ['SLURM_ARRAY_TASK_ID']);r=rows[i]
out=Path(__file__).parent/'runs'/(r['name']+'_'+os.environ['SLURM_ARRAY_JOB_ID'])
cmd=[sys.executable,str(root/r['command'][0]),*r['command'][1:],'--out',str(out)]
print('Running:',cmd,flush=True)
raise SystemExit(subprocess.call(cmd,cwd=root))
''')
    # run_one.py resolves root by a fixed two-level jobs output path. Enforce it.
    if a.out.resolve().parent!=KIT/'plans':
        raise ValueError('Use --out plans/NAME (inside toolkit). No job has been submitted.')
    script=f'''#!/bin/bash
#SBATCH --account=st-zliu-1-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time={a.time}
#SBATCH --array=0-{len(rows)-1}%{a.max_concurrent}
#SBATCH --job-name=xc_formal
#SBATCH --output=xc_formal_%A_%a.out
#SBATCH --error=xc_formal_%A_%a.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
/scratch/st-zliu-1/heqingz/envs/xcarrier/bin/python {shlex.quote(planrel)}/run_one.py
'''
    (a.out/'submit.sbatch').write_text(script)
    print(f'Generated {len(rows)} jobs; concurrency limit {a.max_concurrent}; NOT SUBMITTED.')
    print(f'Read {a.out}/ledger.csv before manually running: sbatch {a.out}/submit.sbatch')

if __name__=='__main__':main()
