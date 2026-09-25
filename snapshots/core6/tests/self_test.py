#!/usr/bin/env python3
"""CPU tests only. No timm/PEFT model construction or radar data access is required."""
from __future__ import annotations
import argparse, copy, importlib.util, json, math, sys, tempfile, types
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from xccontrol.common import KIT,DEFAULT_REPO,load_project,write_json
from xccontrol.recipes import make_config,RECIPES
from xccontrol.objective import build_objective
from xccontrol.dg import per_sample_cosine_head_grads,CosineHeadFishr,ParameterView,snapshot_dict
from xccontrol.metrics import classification
from vendor import swa_utils
from vendor.swad import LossValley
from train_controlled import load_trainable,trainable_state


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,default=DEFAULT_REPO)
    ap.add_argument('--out',type=Path,default=KIT/'local_self_test.json');a=ap.parse_args()
    torch.set_num_threads(1);torch.manual_seed(17)
    report=dict(device='CPU',torch=torch.__version__,tests={},not_tested=['Real DINOv3/LoRA backbone','Real radar image data','GPU training','Actual accuracy or improvement'])
    # Exact analytic per-sample gradient and its differentiable variance, versus autograd.
    z=torch.randn(6,5,dtype=torch.float64,requires_grad=True)
    w=torch.randn(3,5,dtype=torch.float64,requires_grad=True)
    s=torch.randn(3,5,dtype=torch.float64,requires_grad=True)
    y=torch.tensor([0,1,2,0,1,2]);dom=torch.tensor([0,0,0,1,1,1]);scale=2.4;sm=.05
    prior=torch.tensor([-.8,-1.1,-1.7],dtype=torch.float64)
    logits=scale*(F.normalize(z,dim=1)@F.normalize(w,dim=1).t()+
                  F.normalize(z,dim=1)@F.normalize(s,dim=1).t()-.25*F.one_hot(y,3))-prior
    analytic=per_sample_cosine_head_grads(z,{'weight':w,'sensor_weight':s},logits,y,scale,sm)
    losses=F.cross_entropy(logits,y,reduction='none',label_smoothing=sm)
    automatic={k:[] for k in analytic}
    for loss in losses:
        gs=torch.autograd.grad(loss,(w,s),create_graph=True,retain_graph=True)
        for k,g in zip(automatic,gs):automatic[k].append(g.flatten())
    automatic={k:torch.stack(g) for k,g in automatic.items()}
    err=max((analytic[k]-automatic[k]).abs().max().item() for k in analytic)
    torch.testing.assert_close(analytic['weight'],automatic['weight'],rtol=1e-10,atol=1e-10)
    torch.testing.assert_close(analytic['sensor_weight'],automatic['sensor_weight'],rtol=1e-10,atol=1e-10)
    pa,_=CosineHeadFishr(2).penalty(analytic,dom);pb,_=CosineHeadFishr(2).penalty(automatic,dom)
    ga=torch.autograd.grad(pa,(z,w,s),retain_graph=True);gb=torch.autograd.grad(pb,(z,w,s),retain_graph=True)
    grad_error=max((x-y).abs().max().item() for x,y in zip(ga,gb))
    for x1,x2 in zip(ga,gb):torch.testing.assert_close(x1,x2,rtol=1e-9,atol=1e-10)
    report['tests']['cosine_Fishr_vs_autograd']=dict(pass_=True,max_per_sample_grad_error=err,max_penalty_grad_error=grad_error)
    # Restore requires every trainable key: cannot silently leave a random/previous seed weight.
    m=nn.Linear(5,3);state=trainable_state(m)
    with torch.no_grad():m.weight.zero_()
    load_trainable(m,state);torch.testing.assert_close(m.weight,state['weight'])
    try:load_trainable(m,{'weight':state['weight']})
    except RuntimeError:pass
    else:raise AssertionError('Incomplete state was accepted')
    report['tests']['checkpoint_key_integrity']=True
    # Per-class recall != F1 fixture.
    metric=classification([0,0,1,1],[0,0,0,1],['a','b'])
    assert metric['per_class']['a']['recall']==1.0 and abs(metric['per_class']['a']['f1']-.8)<1e-12
    report['tests']['classification_metrics']=True
    # SWAD selection is compared to supplied original code on the same synthetic loss trajectory.
    lib_pkg=types.ModuleType('domainbed.lib');lib_pkg.swa_utils=swa_utils
    old_db=sys.modules.get('domainbed');old_lib=sys.modules.get('domainbed.lib')
    sys.modules['domainbed']=types.ModuleType('domainbed');sys.modules['domainbed.lib']=lib_pkg
    spec=importlib.util.spec_from_file_location('original_swad_test',KIT/'evidence/swad_original.py')
    original_swad=importlib.util.module_from_spec(spec);spec.loader.exec_module(original_swad)
    logger=types.SimpleNamespace(error=lambda *args:None)
    v1=LossValley(types.SimpleNamespace(logger=logger),3,6,.3)
    v2=original_swad.LossValley(types.SimpleNamespace(logger=logger),3,6,.3)
    trajectory=[3.,2.,1.,1.05,1.04,1.03,1.04,1.1,1.2,1.7,2.,2.,2.,2.,2.,2.,2.]
    for ep,loss in enumerate(trajectory):
        test_model=nn.Linear(1,1,bias=False)
        with torch.no_grad():test_model.weight.fill_(float(ep))
        segment=swa_utils.AveragedModel(ParameterView(test_model))
        for st in range(3):segment.update_parameters(ParameterView(test_model),step=ep*3+st)
        v1.update_and_evaluate(copy.deepcopy(segment),0,loss,print)
        v2.update_and_evaluate(copy.deepcopy(segment),0,loss,print)
    old_cuda=nn.Module.cuda
    try:
        nn.Module.cuda=lambda self,*args,**kwargs:self
        final1=v1.get_final_model();final2=v2.get_final_model()
    finally:nn.Module.cuda=old_cuda
    for x1,x2 in zip(final1.parameters(),final2.parameters()):torch.testing.assert_close(x1,x2,rtol=0,atol=0)
    assert v1.dead_valley==v2.dead_valley and v1.converge_step==v2.converge_step
    report['tests']['SWAD_matches_supplied_original']=dict(pass_=True,converged=v1.is_converged,dead_valley=v1.dead_valley)
    if old_db is None:sys.modules.pop('domainbed',None)
    else:sys.modules['domainbed']=old_db
    if old_lib is None:sys.modules.pop('domainbed.lib',None)
    else:sys.modules['domainbed.lib']=old_lib
    # Import actual supplied model/loss code. Replace ONLY encoder factory in this test.
    cfg,lib,trainer=load_project(a.repo.resolve(),make_config('proposed','original7','77GHz'))
    objective,text=build_objective(trainer)
    class ToyEncoder(nn.Module):
        def __init__(self):super().__init__();self.fc=nn.Linear(3,16)
        def forward(self,x):return self.fc(x.mean((-2,-1)))
    lib._create_feature_encoder=lambda name:ToyEncoder()
    checked=[]
    for recipe in RECIPES:
        for k,v in make_config(recipe,'original7','77GHz').items():setattr(cfg,k,v)
        cfg.HEAD_HIDDEN=8;cfg.FEATURE_DROPOUT=0.;cfg.IMG_SIZE=8
        cfg.CLASSES=['a','b','c'];cfg.NUM_CLASSES=3
        model=lib.TimmBackboneV921('toy',3,adapter_mode='full_ft')
        miro=lib.MIROProjector(8,16)
        adv=lib.CarrierAdversary(8,4,3 if cfg.V13_GRL_DISCRETE else 1) if cfg.V13_GRL_WEIGHT else None
        dc=lib.DomainClassifier(8,hidden=4,num_domains=2) if cfg.USE_DANN else None
        x=torch.randn(6,3,8,8);aux=torch.randn_like(x);r=torch.randn(6)
        prior=torch.log(torch.ones(3)/3)
        glog=torch.log(torch.tensor([10.,24.])) if adv else None
        edges=torch.linspace(*lib.das_log_carrier_range(),4)[1:-1] if cfg.V13_GRL_DISCRETE else None
        inputs=dict(model=model,miro=miro,domain_clf=dc,carrier_adv=adv,grl_band_log_ghz=glog,
                    grl_bin_edges=edges,grl_discrete=cfg.V13_GRL_DISCRETE,grl_bins=3,source_loader=None,
                    x_src=x,y_src=y,d_src=dom,x_aux=aux,r_src=r,global_step=17,total_steps=100,
                    logit_prior=prior,ce=nn.CrossEntropyLoss(label_smoothing=.05),ce_domain=nn.CrossEntropyLoss())
        loss,parts,zout,adj,raw=objective(**inputs)
        assert torch.isfinite(loss)
        loss.backward()
        assert model.neck[2].weight.grad is not None and torch.isfinite(model.neck[2].weight.grad).all()
        assert float(parts['v15'].detach())==0. and float(parts['v15r'].detach())==0.
        checked.append(recipe)
    report['tests']['all_14_recipe_forward_backward_with_toy_encoder']=checked
    report['status']='PASS_CPU_ONLY'
    write_json(a.out,report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
