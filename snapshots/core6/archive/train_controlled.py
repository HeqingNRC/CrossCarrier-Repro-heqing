#!/usr/bin/env python3
"""Target-blind controller around the audited source objective/model/augmentation.

Does NOT claim a byte-identical replay of the original target-monitored controller.
No target image is decoded until training, checkpoint selection, and SWAD finalize.
No seven-class fine-tuned checkpoint is used to initialize a six-class experiment.
"""
from __future__ import annotations
import argparse, contextlib, importlib.metadata, json, logging, math, os, platform, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from xccontrol.common import *
from xccontrol.recipes import RECIPES, make_config
from xccontrol.objective import build_objective, ARGS
from xccontrol.dg import CosineHeadFishr, ParameterView, snapshot_dict, per_sample_cosine_head_grads
from xccontrol.metrics import classification, write_predictions
from vendor.swa_utils import AveragedModel
from vendor.swad import LossValley


def args_parser():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo',type=Path,default=DEFAULT_REPO)
    ap.add_argument('--prepared',type=Path,default=KIT/'prepared')
    ap.add_argument('--dataset',choices=['original7','core6'],required=True)
    ap.add_argument('--target',choices=FREQUENCIES,default='77GHz')
    ap.add_argument('--recipe',choices=list(RECIPES),default='proposed')
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--epochs',type=int,default=100)
    ap.add_argument('--batch',type=int,default=16)
    ap.add_argument('--eval-batch',type=int,default=32)
    ap.add_argument('--precision',choices=['bf16','fp32'],default='bf16')
    ap.add_argument('--smoke-steps',type=int,default=0,help='Stop after this many optimizer steps; NO target evaluation')
    ap.add_argument('--acknowledge-duplicate-risk',action='store_true')
    ap.add_argument('--swad',action='store_true')
    ap.add_argument('--source-only',action='store_true',help='Hyperparameter pilot: save selected weights/source metrics, never load target images')
    ap.add_argument('--fishr',action='store_true')
    ap.add_argument('--fishr-lambda',type=float,default=None,help='Explicit candidate required with --fishr; old reference 1000 is NOT assumed suitable')
    ap.add_argument('--fishr-anneal',type=int,default=1500)
    ap.add_argument('--fishr-ema',type=float,default=0.95)
    return ap


def get_amp(precision):
    return (lambda: torch.autocast('cuda',dtype=torch.bfloat16)) if precision=='bf16' else contextlib.nullcontext


def trainable_state(model):
    return {n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}


def load_trainable(model,state):
    """Frozen base is intentionally absent, but every trainable key must match."""
    expected={n:p for n,p in model.named_parameters() if p.requires_grad}
    if set(state)!=set(expected):
        raise RuntimeError(f'Trainable key mismatch: missing={sorted(set(expected)-set(state))}; extra={sorted(set(state)-set(expected))}')
    with torch.no_grad():
        for n,p in expected.items():
            if tuple(p.shape)!=tuple(state[n].shape): raise ValueError(f'Wrong shape: {n}')
            p.copy_(state[n].to(p.device,p.dtype))


def evaluate(model,loader,rows,classes,amp):
    model.eval(); all_logits=[]; all_y=[]
    with torch.no_grad():
        for x,y,_d in loader:
            with amp():
                z=model.encode_adapted(x)
                logits=model.logits_from_neck(z,margin=False)
            all_logits.append(logits.float().cpu()); all_y.append(y.cpu())
    logits=torch.cat(all_logits); labels=torch.cat(all_y)
    expected=torch.tensor([int(r['class_idx_7c']) for r in rows])
    if not torch.equal(labels,expected):raise RuntimeError('Prediction order does not match manifest.')
    ce=F.cross_entropy(logits,labels,reduction='none').numpy()
    freqs=np.asarray([r['frequency'] for r in rows])
    # Equal-domain mean source validation CE/accuracy. Never contains target images.
    dloss={f:float(ce[freqs==f].mean()) for f in sorted(set(freqs))}
    preds=logits.argmax(1).numpy(); yy=labels.numpy()
    dacc={f:float((preds[freqs==f]==yy[freqs==f]).mean()) for f in sorted(set(freqs))}
    result=classification(yy,preds,classes)
    result.update(equal_domain_ce=float(np.mean(list(dloss.values()))),
                  equal_domain_accuracy=float(np.mean(list(dacc.values()))),per_domain_ce=dloss,per_domain_accuracy=dacc)
    return result,yy,logits.numpy()


def save_model(path,model_state,args,protocol,epoch,step,selection,extra=None):
    payload=dict(format='xc_controlled_trainable_v1',state=model_state,classes=protocol['classes'],
                 dataset=args.dataset,target=args.target,recipe=args.recipe,seed=args.seed,
                 backbone='vit_large_patch16_dinov3.lvd1689m',epoch=epoch,step=step,
                 selection=selection,smoke=bool(args.smoke_steps),precision=args.precision,
                 protocol_manifest_hashes=protocol['manifest_hashes'])
    if extra:payload.update(extra)
    tmp=path.with_suffix(path.suffix+'.tmp');torch.save(payload,tmp);tmp.replace(path)


def main():
    a=args_parser().parse_args()
    a.repo=a.repo.resolve();a.prepared=a.prepared.resolve();a.out=a.out.resolve()
    if a.fishr and a.fishr_lambda is None:raise ValueError('Specify --fishr-lambda explicitly; evaluate candidate settings on source validation only.')
    if a.epochs<5 and not a.smoke_steps: raise ValueError('EMA starts at epoch 5; formal run needs >=5 epochs.')
    if a.fishr and (a.fishr_lambda<0 or a.fishr_anneal<0 or not 0<=a.fishr_ema<1):raise ValueError('Invalid Fishr settings')
    if a.batch<2 or a.eval_batch<1 or a.smoke_steps<0:raise ValueError('Invalid batch/step arguments')
    if a.out.exists():raise FileExistsError(f'Refusing to overwrite {a.out}. Choose a NEW run directory.')
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required. Submit with sbatch; do not train on a login node.')
    folder,protocol=read_protocol(a.prepared,a.dataset,a.target,bool(a.smoke_steps),a.acknowledge_duplicate_risk)
    cfg=make_config(a.recipe,a.dataset,a.target,a.epochs,a.batch,a.precision)
    cfg['EVAL_BATCH']=a.eval_batch
    config,lib,original=load_project(a.repo.resolve(),cfg)
    objective,objective_text=build_objective(original)
    amp=get_amp(a.precision)
    # Check requested precision explicitly; never silently switch dtype.
    with amp():
        probe=torch.randn(8,8,device='cuda'); _=probe@probe
    lib.set_seed(a.seed)
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','4')))
    device=torch.device('cuda')
    a.out.mkdir(parents=True); (a.out/'checkpoints').mkdir(); (a.out/'predictions').mkdir()
    log_file=(a.out/'train.log').open('w',encoding='utf-8')
    def log(msg):
        text=f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}'
        print(text,flush=True);log_file.write(text+'\n');log_file.flush()
    runtime={p:importlib.metadata.version(p) for p in ['torch','torchvision','timm','peft','transformers','accelerate','numpy','pandas','Pillow','safetensors']}
    write_json(a.out/'effective_config.json',serializable_config(config))
    write_json(a.out/'run_manifest.json',dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        protocol=protocol,packages=runtime,python=sys.version,platform=platform.platform(),gpu=torch.cuda.get_device_name(0),
        cuda_build=torch.version.cuda,scientific_status='SMOKE_ONLY' if a.smoke_steps else 'NEW_TRAINING_NOT_YET_EVALUATED',
        controller='target-blind; reuses audited source forward/loss AST; not identical original controller/RNG trajectory',
        unknown_original_launch_flags=['SKIP_ORACLE','original per-epoch diagnostic evaluation cadence'],
        fishr_variant='both cosine-classifier weight gradients of adjusted, smoothed CE; real source domains; missing-domain batches skipped' if a.fishr else None,
        source_validation_selection='equal-domain mean kinematic-head CE for SWAD; source-best EMA by equal-domain accuracy',
        training_initialization='local pretrained backbone + newly initialized LoRA/heads, NOT shipped final-EMA weights'))
    (a.out/'audited_objective_used.py').write_text(objective_text)
    (a.out/'pip_freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    # Only source manifests/images are loaded here. test.csv is not opened until after training.
    tr_rows=read_csv(folder/'train.csv');val_rows=read_csv(folder/'val.csv')
    train_df=pd.DataFrame(tr_rows);val_df=pd.DataFrame(val_rows)
    for df in (train_df,val_df):
        df['class_idx_7c']=df['class_idx_7c'].astype(int);df['freq_idx']=df['freq_idx'].astype(int)
    source_loader=lib.GpuSourceProvider(train_df,a.seed,device,log=log)
    val_loader=lib.build_gpu_eval_loader(val_df,device,batch=a.eval_batch)
    if len(source_loader)<1:raise ValueError('Training set smaller than batch.')
    model=lib.TimmBackboneV921(config.DEFAULT_BACKBONE,len(protocol['classes']),adapter_mode='lora').to(device)
    miro=lib.MIROProjector(config.HEAD_HIDDEN,model.enc_dim).to(device)
    domain_clf=lib.DomainClassifier(config.HEAD_HIDDEN,hidden=config.DANN_HIDDEN,num_domains=config.NUM_FREQ_DOMAINS).to(device) if config.USE_DANN else None
    carrier_adv=None; grl_logs=None; grl_edges=None
    if config.V13_GRL_WEIGHT>0:
        carrier_adv=lib.CarrierAdversary(config.HEAD_HIDDEN,hidden=config.V13_GRL_HIDDEN,
                    out_dim=config.V13_GRL_BINS if config.V13_GRL_DISCRETE else 1).to(device)
        grl_logs=torch.tensor([math.log(lib.parse_freq_ghz(f)) for f in config.TRAIN_FREQS],device=device)
        if config.V13_GRL_DISCRETE:
            lo,hi=lib.das_log_carrier_range();grl_edges=torch.linspace(lo,hi,config.V13_GRL_BINS+1,device=device)[1:-1].contiguous()
    parameters=[p for p in model.parameters() if p.requires_grad]+list(miro.parameters())
    for mod in (domain_clf,carrier_adv):
        if mod is not None: parameters+=list(mod.parameters())
    opt=torch.optim.AdamW(parameters,lr=config.LR,weight_decay=config.WEIGHT_DECAY)
    ema=lib.ModelEMA(model,config.EMA_DECAY)  # Preserve source: initialize before training, start updates at epoch 5.
    prior=lib.compute_logit_prior(train_df,len(protocol['classes']),device)
    ce=torch.nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING);ce_domain=torch.nn.CrossEntropyLoss()
    live_view=ParameterView(model)
    segment=AveragedModel(live_view,device=device) if a.swad else None
    valley=LossValley(type('Evaluator',(),{'logger':logging.getLogger('SWAD')})(),3,6,0.3) if a.swad else None
    fishr=CosineHeadFishr(config.NUM_FREQ_DOMAINS,a.fishr_ema) if a.fishr else None
    counts=dict(model_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                full_optimizer_trainable=sum(p.numel() for p in parameters),
                inference_trainable=sum(p.numel() for n,p in model.named_parameters() if p.requires_grad and
                                        (n.startswith('encoder.') or n.startswith('neck.') or n=='weight')))
    write_json(a.out/'parameter_counts.json',counts)
    structure=dict(
        encoder_dimension=model.enc_dim,
        lora_modules=[dict(name=n,kind=type(m).__name__) for n,m in model.named_modules() if hasattr(m,'lora_A')],
        trainable_parameters=[dict(name=n,shape=list(p.shape),numel=p.numel()) for n,p in model.named_parameters() if p.requires_grad],
        inference_parameter_filter='encoder.* + neck.* + weight; reference/frequency/sensor/adversary/projector excluded',
    )
    write_json(a.out/'model_structure.json',structure)
    log(f'[start] {a.dataset} {config.TRAIN_FREQS}->{a.target} {a.recipe} seed={a.seed}; train={len(train_df)} val={len(val_df)}; target NOT loaded')
    log(f'[parameters] {counts}; precision={a.precision}; Fishr={a.fishr}; SWAD={a.swad}')
    log(f'[GPU] {torch.cuda.get_device_name(0)}; compute_capability={torch.cuda.get_device_capability()}; bf16_native={torch.cuda.is_bf16_supported(including_emulation=False)}; bf16_tensor_supported={torch.cuda.is_bf16_supported()}')
    total_steps=a.epochs*len(source_loader);warmup_steps=config.WARMUP_EPOCHS*len(source_loader)
    step=0;best_acc=-1.0;best_state=None;history=[];first_grad_report=None
    start_time=time.time()
    for ep in range(1,a.epochs+1):
        epoch_start=time.time(); source_loader.set_epoch(ep)
        model.train();model.set_encoder_train_mode();miro.train()
        for mod in (domain_clf,carrier_adv):
            if mod is not None:mod.train()
        sums={}; nb=0
        for _ in range(len(source_loader)):
            lr=original.cosine_lr_schedule(step,total_steps,warmup_steps,config.LR)
            if fishr and step==a.fishr_anneal and a.fishr_anneal>0:
                # Preserve the original Fishr convention when its regularizer turns on.
                opt=torch.optim.AdamW(parameters,lr=lr,weight_decay=config.WEIGHT_DECAY)
                log(f'[fishr] reset Adam at anneal step {step}')
            for group in opt.param_groups:group['lr']=lr
            x,y,d,aux,residual=next(source_loader)
            opt.zero_grad(set_to_none=True)
            inputs=dict(model=model,miro=miro,domain_clf=domain_clf,carrier_adv=carrier_adv,
                        grl_band_log_ghz=grl_logs,grl_bin_edges=grl_edges,grl_discrete=config.V13_GRL_DISCRETE,
                        grl_bins=config.V13_GRL_BINS,source_loader=source_loader,x_src=x,y_src=y,d_src=d,
                        x_aux=aux,r_src=residual,global_step=step,total_steps=total_steps,
                        logit_prior=prior,ce=ce,ce_domain=ce_domain)
            with amp():
                loss,parts,z,adjusted,raw=objective(**inputs)
            if fishr:
                # Penalty derivative computed analytically in float32 to avoid routing
                # high-order autograd through V100 fused attention. Main CE remains unchanged.
                with torch.autocast('cuda',enabled=False):
                    z32=z.float()
                    logits32=model.total_logits_from_neck(z32,y,margin=True)-config.LOGIT_ADJUST_TAU*prior
                    grads=per_sample_cosine_head_grads(z32,{'weight':model.weight,'sensor_weight':model.sensor_weight},
                                logits32,y,config.ARC_SCALE,config.LABEL_SMOOTHING)
                    penalty,domain_counts=fishr.penalty(grads,d)
                    fw=a.fishr_lambda if step>=a.fishr_anneal else 0.0
                    loss=loss+fw*penalty
                    parts.update(fishr_penalty=penalty,fishr_weight=fw)
            if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss at step {step}')
            loss.backward()
            if first_grad_report is None or (step==1 and a.smoke_steps):
                # Check nonzero finite gradients by branch. At step0 the scheduled GRL lambda is zero.
                report={}
                for prefix in ('encoder.','neck.','freq_neck.','freq_head.','weight','sensor_weight'):
                    ps=[p for n,p in model.named_parameters() if n.startswith(prefix) and p.requires_grad]
                    active=[p.grad for p in ps if p.grad is not None]
                    report[prefix]=dict(parameter_tensors=len(ps),gradient_tensors=len(active),
                        finite=all(bool(torch.isfinite(g).all()) for g in active),
                        nonzero_tensors=sum(int(bool((g!=0).any())) for g in active))
                if any(not r['finite'] for r in report.values()):raise FloatingPointError('Nonfinite parameter gradient')
                first_grad_report=report;write_json(a.out/f'gradient_check_step{step}.json',report)
            opt.step()
            if ep>=config.EMA_START_EPOCH:ema.update(model)
            if segment is not None and not valley.dead_valley:segment.update_parameters(live_view,step=step)
            for k,v in dict(parts,total=loss).items():
                sums[k]=sums.get(k,0.0)+float(v.detach() if torch.is_tensor(v) else v)
            nb+=1;step+=1
            if step%10==0 or a.smoke_steps:
                log(f'[train] ep={ep} step={step}/{total_steps} loss={float(loss.detach()):.5f} lr={lr:.3g}')
            if a.smoke_steps and step>=a.smoke_steps:break
        # Forward pass on source validation only. Evaluation uses the deployed kinematic head.
        val_live,val_live_y,val_live_logits=evaluate(model,val_loader,val_rows,protocol['classes'],amp)
        val_ema=None
        if ep>=config.EMA_START_EPOCH:
            with lib.SwapEMA(model,ema):val_ema,_,_=evaluate(model,val_loader,val_rows,protocol['classes'],amp)
            if val_ema['equal_domain_accuracy']>best_acc:
                best_acc=val_ema['equal_domain_accuracy'];best_state={k:v.cpu().clone() for k,v in ema.state_dict().items()}
                save_model(a.out/'checkpoints/source_best_ema.pt',best_state,a,protocol,ep,step,'source_equal_domain_accuracy')
        if valley is not None and not valley.dead_valley:
            valley.update_and_evaluate(segment,val_live['equal_domain_accuracy'],val_live['equal_domain_ce'],log)
            segment=AveragedModel(live_view,device=device)
        record=dict(epoch=ep,step=step,seconds=time.time()-epoch_start,losses={k:v/nb for k,v in sums.items()},
                    das=source_loader.current_das_stage(),source_validation_live=val_live,source_validation_ema=val_ema,
                    target_evaluated=False)
        history.append(record);write_json(a.out/'history.json',history)
        log(f"[epoch] {ep} seconds={record['seconds']:.1f} source_F1={val_live['macro_f1']:.4f} target=NOT_EVALUATED")
        if a.smoke_steps:
            state=trainable_state(model)
            save_model(a.out/'checkpoints/smoke_live.pt',state,a,protocol,ep,step,'SMOKE_ONLY')
            with torch.no_grad():
                # Exercise restoration, including actual disk serialization, without trusting strict=False.
                next(p for p in model.parameters() if p.requires_grad).add_(0.123)
                load_trainable(model,torch.load(a.out/'checkpoints/smoke_live.pt',map_location='cpu',weights_only=True)['state'])
                after,yy,ll=evaluate(model,val_loader,val_rows,protocol['classes'],amp)
            roundtrip_error=float(np.max(np.abs(ll-val_live_logits)))
            if roundtrip_error>1e-5 or not np.array_equal(yy,val_live_y):raise RuntimeError(f'Save/reload changed logits: {roundtrip_error}')
            ema_test=lib.ModelEMA(model,config.EMA_DECAY);ema_test.update(model)
            with lib.SwapEMA(model,ema_test):pass
            write_json(a.out/'SMOKE_PASS.json',dict(status='PASS',optimizer_steps=step,loss_finite=True,
                       gradient_checks_saved=True,checkpoint_roundtrip=True,roundtrip_max_logit_difference=roundtrip_error,ema_api_exercised=True,
                       actual_formal_ema_updates=(ep>=config.EMA_START_EPOCH),target_evaluated=False,
                       elapsed_seconds=time.time()-start_time,scientific_results=False))
            log('[PASS] GPU SMOKE ONLY. No target score, no formal result, no ablation claim.');log_file.close();return
    # The final EMA is ALWAYS saved; no source >=90% gate and no target metric gate.
    final_state={k:v.cpu().clone() for k,v in ema.state_dict().items()}
    save_model(a.out/'checkpoints/final_ema.pt',final_state,a,protocol,a.epochs,step,'fixed_final_ema')
    selected=[('final_ema',final_state)]
    if best_state is not None:selected.append(('source_best_ema',best_state))
    swad_info=None
    if valley is not None:
        avg=valley.get_final_model()
        swad_info=dict(converged=valley.is_converged,dead_valley=valley.dead_valley,
                       start_step=avg.start_step,end_step=avg.end_step,n_segments=int(avg.n_averaged),
                       threshold=valley.threshold,source_metric='equal-domain mean live kinematic-head CE',
                       crosses_fishr_anneal=bool(a.fishr and avg.start_step<=a.fishr_anneal<=avg.end_step))
        swad_state=snapshot_dict(avg)
        selector='swad' if valley.is_converged else 'swad_not_converged_last_segment_fallback'
        save_model(a.out/'checkpoints/swad.pt',swad_state,a,protocol,a.epochs,step,selector,swad_info)
        selected.append((selector,swad_state))
    # Lock the selected model states before any target image is decoded.
    write_json(a.out/'selection_frozen_before_target.json',dict(selectors=[s for s,_ in selected],swad=swad_info,
                checkpoint_sha256={p.name:sha256(p) for p in (a.out/'checkpoints').glob('*.pt')},
                fishr=dict(updates=fishr.updates,skipped=fishr.skipped) if fishr else None))
    source_results={}
    for selector,state in selected:
        load_trainable(model,state)
        source_results[selector]=evaluate(model,val_loader,val_rows,protocol['classes'],amp)[0]
    write_json(a.out/'source_validation_final.json',source_results)
    if a.source_only:
        write_json(a.out/'SOURCE_ONLY_DONE.json',dict(status='SOURCE_ONLY_COMPLETE',target_evaluated=False,
                   elapsed_seconds=time.time()-start_time,source_validation=source_results))
        log('[done] SOURCE-ONLY pilot. Select settings using source validation, then explicitly evaluate locked checkpoints.')
        log_file.close();return
    test_rows=read_csv(folder/'test.csv');test_df=pd.DataFrame(test_rows)
    for key in ('class_idx_7c','freq_idx'):test_df[key]=test_df[key].astype(int)
    test_loader=lib.build_gpu_eval_loader(test_df,device,batch=a.eval_batch)
    results={}
    for selector,state in selected:
        load_trainable(model,state)
        _r,y,logits=evaluate(model,test_loader,test_rows,protocol['classes'],amp)
        results[selector]=write_predictions(a.out/'predictions'/selector,test_rows,y,logits,protocol['classes'],
                     dict(selection=selector,seed=a.seed,task=a.dataset,target=a.target,recipe=a.recipe))
        log(f"[target] selector={selector} n={len(y)} macro-F1={results[selector]['macro_f1']:.6f}; acc={results[selector]['accuracy']:.6f}")
    write_json(a.out/'DONE.json',dict(status='COMPLETE',results=results,elapsed_seconds=time.time()-start_time,
                caveat='One new training run. Aggregate the prespecified seed set before comparison; no improvement is assumed.'))
    log_file.close()

if __name__=='__main__':main()
