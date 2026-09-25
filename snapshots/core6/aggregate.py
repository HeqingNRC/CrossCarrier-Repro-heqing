#!/usr/bin/env python3
"""Aggregate fixed trained seeds. Never chooses a model using target scores."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from xccontrol.common import write_json
from xccontrol.metrics import classification,stratified_bootstrap


def load_runs(runs,selector):
    arrays=[];seeds=[];signatures=[]
    for run in runs:
        if (run/'SMOKE_PASS.json').exists():raise RuntimeError(f'Smoke run cannot be scored: {run}')
        if not ((run/'DONE.json').exists() or (run/'REPLAY_DONE.json').exists()):raise RuntimeError(f'Incomplete run: {run}')
        if (run/'run_manifest.json').exists():
            meta=json.loads((run/'run_manifest.json').read_text());a=meta['arguments']
            signatures.append({k:a[k] for k in ('dataset','target','recipe','epochs','batch','precision','fishr','fishr_lambda','fishr_anneal','fishr_ema','swad')})
            seeds.append(int(a['seed']))
        else:
            m=json.loads((run/'replay_provenance.json').read_text())
            signatures.append(dict(legacy=True,recipe=m['recipe'],protocol=m['protocol']['manifest_hashes']))
            seeds.append(m['seed'])
        with np.load(run/f'predictions/{selector}.npz',allow_pickle=False) as z:
            arrays.append({k:z[k].copy() for k in z.files})
    if len(set(seeds))!=len(seeds):raise ValueError('Duplicate random seed runs supplied.')
    if any(s!=signatures[0] for s in signatures):raise ValueError('Runs differ in method, precision, budget or protocol; do not ensemble across experiments.')
    first=arrays[0]
    for a in arrays[1:]:
        for key in ('paths','labels','classes'):
            if not np.array_equal(first[key],a[key]):raise ValueError(f'{key} differ across runs.')
    return np.stack([a['logits'] for a in arrays]),first['labels'],first['classes'].tolist(),first['paths'],seeds


def ensemble_predictions(L):
    pp=L.argmax(2);mean_logits=L.mean(0)
    e=np.exp(L-L.max(2,keepdims=True));prob=e/e.sum(2,keepdims=True)
    majority=[]
    for j in range(L.shape[1]):
        vals,counts=np.unique(pp[:,j],return_counts=True); winners=vals[counts==counts.max()]
        majority.append(int(winners[0]) if len(winners)==1 else int(mean_logits[j].argmax()))
    return dict(majority=np.asarray(majority),logit_average=mean_logits.argmax(1),posterior_average=prob.mean(0).argmax(1))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runs',nargs='+',type=Path,required=True)
    ap.add_argument('--selector',default='final_ema')
    ap.add_argument('--n-boot',type=int,default=2000)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    if len(a.runs)<2:raise ValueError('Supply at least two distinct seeds for ensemble aggregation.')
    L,y,classes,paths,seeds=load_runs(a.runs,a.selector)
    pred=L.argmax(2);per=[classification(y,p,classes) for p in pred]
    f1=np.asarray([r['macro_f1'] for r in per]);acc=np.asarray([r['accuracy'] for r in per])
    ens=ensemble_predictions(L)
    r=dict(seeds=seeds,classes=classes,n=len(y),selector=a.selector,
           single_macro_f1_mean=float(f1.mean()),single_macro_f1_std_population=float(f1.std(ddof=0)),
           single_macro_f1_std_sample=float(f1.std(ddof=1)),single_accuracy_mean=float(acc.mean()),
           single_accuracy_std_population=float(acc.std(ddof=0)),per_seed={str(s):p for s,p in zip(seeds,per)},
           ensemble={k:classification(y,p,classes) for k,p in ens.items()},
           note='Population std reproduces original report convention. Bootstrap is image-conditional, not a confidence interval over training seeds.')
    for i,s in enumerate(seeds):r['per_seed'][str(s)]['bootstrap']=stratified_bootstrap(y,pred[i],classes,a.n_boot)
    for k,p in ens.items():r['ensemble'][k]['bootstrap']=stratified_bootstrap(y,p,classes,a.n_boot)
    # Image-bootstrap CI for the mean of these fixed seed-model F1s (not first-seed CI).
    rng=np.random.default_rng(20260913);groups=[np.flatnonzero(y==i) for i in range(len(classes))]
    vals=[]
    for _ in range(a.n_boot):
        ix=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])
        vals.append(np.mean([classification(y[ix],p[ix],classes)['macro_f1'] for p in pred]))
    r['fixed_seed_mean_image_bootstrap_ci95']=np.quantile(vals,[.025,.975]).tolist()
    a.out.mkdir(parents=True);write_json(a.out/'metrics.json',r)
    np.savez_compressed(a.out/'predictions.npz',paths=paths,labels=y,classes=np.asarray(classes),
                        per_seed_logits=L,seeds=np.asarray(seeds),**ens)
    lines=['# Training reproduction results','',f'Seeds: {seeds}; n={len(y)}; selector={a.selector}',
           '',f'Single macro-F1: {f1.mean():.6f} ± {f1.std():.6f} (population SD)',
           '', '| Rule | Macro-F1 | Accuracy | Image bootstrap 95% CI |','|---|---:|---:|---|']
    for k,v in r['ensemble'].items():lines.append(f"| {k} | {v['macro_f1']:.6f} | {v['accuracy']:.6f} | {v['bootstrap']['ci95']} |")
    lines+=['','CI ignores participant/session clustering and duplicate dependence. No target-based checkpoint selection is performed here.']
    (a.out/'metrics.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

if __name__=='__main__':main()
