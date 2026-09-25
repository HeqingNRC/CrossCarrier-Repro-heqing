from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
from .common import write_json


def classification(y,pred,classes):
    y=np.asarray(y,dtype=int); pred=np.asarray(pred,dtype=int); n=len(classes)
    if y.shape!=pred.shape or y.ndim!=1 or len(y)==0: raise ValueError('Invalid/empty prediction arrays')
    if y.min()<0 or pred.min()<0 or y.max()>=n or pred.max()>=n: raise ValueError('Label outside class map')
    cm=np.bincount(n*y+pred,minlength=n*n).reshape(n,n)
    tp=np.diag(cm); support=cm.sum(1); predicted=cm.sum(0)
    precision=np.divide(tp,predicted,out=np.zeros(n),where=predicted!=0)
    recall=np.divide(tp,support,out=np.zeros(n),where=support!=0)
    f1=np.divide(2*tp,support+predicted,out=np.zeros(n),where=(support+predicted)!=0)
    return dict(n=len(y),accuracy=float(tp.sum()/len(y)),macro_f1=float(f1.mean()),
                balanced_accuracy=float(recall.mean()),confusion=cm.tolist(),
                per_class={c:dict(precision=float(precision[i]),recall=float(recall[i]),f1=float(f1[i]),
                                  support=int(support[i])) for i,c in enumerate(classes)})


def stratified_bootstrap(y,pred,classes,n_boot=2000,seed=20260913):
    y=np.asarray(y); pred=np.asarray(pred); rng=np.random.default_rng(seed)
    groups=[np.flatnonzero(y==i) for i in range(len(classes))]
    if any(len(g)==0 for g in groups): raise ValueError('All classes must be present for this bootstrap.')
    values=[]
    for _ in range(n_boot):
        ix=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])
        values.append(classification(y[ix],pred[ix],classes)['macro_f1'])
    return dict(ci95=np.quantile(values,[.025,.975]).tolist(),resamples=n_boot,seed=seed,
                estimator='macro-F1 of these fixed predictions',unit='class-stratified images',
                limitation='Conditional on trained weights; ignores subject/session clustering and duplicate dependence.')


def write_predictions(path,rows,y,logits,classes,extra=None):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    logits=np.asarray(logits,dtype=np.float32); y=np.asarray(y,dtype=np.int64)
    if logits.shape!=(len(rows),len(classes)): raise ValueError('Output/class shape mismatch')
    ids=np.asarray([r['path'] for r in rows],dtype=str)
    np.savez_compressed(path.with_suffix('.npz'),paths=ids,labels=y,logits=logits,classes=np.asarray(classes))
    pred=logits.argmax(1)
    with path.with_suffix('.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['path','frequency','y_true','y_pred','true_class','predicted_class'])
        for row,yy,pp in zip(rows,y,pred):w.writerow([row['path'],row['frequency'],yy,pp,classes[yy],classes[pp]])
    report=classification(y,pred,classes)
    if extra: report.update(extra)
    write_json(path.with_suffix('.json'),report)
    return report
