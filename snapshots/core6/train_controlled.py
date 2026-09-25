#!/usr/bin/env python3
"""New Core6 target-blind controller; reuses the recovered, hash-pinned model,
GPU augmentation, and original forward/loss block. No old task checkpoint used.
FP16 uses GradScaler. This is not a bitwise replay of a historical GPU run.
"""
from __future__ import annotations
import argparse, contextlib, importlib.metadata, json, math, os, platform, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from xccontrol.common import (KIT, DEFAULT_REPO, CLASSES6, FREQUENCIES, read_csv,
                              sha256, write_json, load_project, serializable_config)
from xccontrol.recipes import make_config
from xccontrol.objective import build_objective
from xccontrol.metrics import classification, write_predictions
from suite_constants import RECIPES, EXPECTED_N


def get_amp(precision):
    if precision == 'fp16':
        return lambda: torch.autocast('cuda', dtype=torch.float16)
    if precision == 'bf16':
        return lambda: torch.autocast('cuda', dtype=torch.bfloat16)
    if precision == 'fp32':
        return contextlib.nullcontext
    raise ValueError(precision)


def trainable_state(model):
    return {n: p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}


def load_trainable(model, state):
    expected = {n:p for n,p in model.named_parameters() if p.requires_grad}
    if set(state) != set(expected):
        raise RuntimeError(f'Checkpoint trainable keys differ: missing={set(expected)-set(state)}, extra={set(state)-set(expected)}')
    with torch.no_grad():
        for n,p in expected.items():
            if p.shape != state[n].shape:
                raise RuntimeError(f'Checkpoint shape differs: {n}')
            if not torch.isfinite(state[n]).all():
                raise FloatingPointError(f'Non-finite checkpoint: {n}')
            p.copy_(state[n].to(device=p.device, dtype=p.dtype))


def evaluate(model, loader, rows, classes, amp):
    model.eval()
    logits_all, labels_all = [], []
    with torch.no_grad():
        for x,y,_ in loader:
            with amp():
                z = model.encode_adapted(x)
                logits = model.logits_from_neck(z, margin=False)
            if not torch.isfinite(logits).all():
                raise FloatingPointError('Non-finite evaluation logits')
            logits_all.append(logits.float().cpu())
            labels_all.append(y.cpu())
    logits = torch.cat(logits_all).numpy()
    labels = torch.cat(labels_all).numpy()
    expected = np.array([int(r['class_idx_7c']) for r in rows])
    if not np.array_equal(labels, expected):
        raise RuntimeError('Loader order differs from manifest order')
    metrics = classification(labels, logits.argmax(1), classes)
    return metrics, labels, logits


def load_rows(folder, split, protocol, verify_images=True):
    p = folder / f'{split}.csv'
    if sha256(p) != protocol['manifest_hashes'][p.name]:
        raise RuntimeError(f'Prepared manifest changed: {p}')
    rows = read_csv(p)
    allowed = [protocol['target']] if split == 'test' else protocol['sources']
    if len(rows) != EXPECTED_N[protocol['target']][split]:
        raise RuntimeError(f'Unexpected {split} count: {len(rows)}')
    for r in rows:
        if r['frequency'] not in allowed:
            raise RuntimeError(f'Wrong carrier in {split}: {r}')
        if r['class'] not in CLASSES6 or int(r['class_idx_7c']) != CLASSES6.index(r['class']):
            raise RuntimeError(f'Wrong six-class mapping: {r}')
        if verify_images and sha256(Path(r['path'])) != r['file_sha256']:
            raise RuntimeError(f'Image changed since preparation: {r["path"]}')
    df = pd.DataFrame(rows)
    for col in ('class_idx_7c', 'freq_idx'):
        df[col] = df[col].astype(int)
    return rows, df


def save_checkpoint(path, state, a, protocol, epoch, updates, skipped):
    payload = dict(format='core6_fixed_final_ema_v2', state=state, epoch=epoch,
                   seed=a.seed, recipe=a.recipe, dataset='core6', target=a.target,
                   classes=CLASSES6, precision=a.precision,
                   selection='fixed_epoch100_ema' if not a.smoke_steps else 'SMOKE_ONLY',
                   smoke=bool(a.smoke_steps), optimizer_updates=updates, skipped_steps=skipped,
                   source_hashes=protocol['source_hashes'],
                   manifest_hashes=protocol['manifest_hashes'])
    tmp = path.with_name(path.name + '.tmp')
    torch.save(payload, tmp)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', type=Path, default=DEFAULT_REPO)
    ap.add_argument('--prepared', type=Path, default=KIT/'prepared')
    ap.add_argument('--dataset', choices=['core6'], default='core6')
    ap.add_argument('--target', choices=FREQUENCIES, required=True)
    ap.add_argument('--recipe', choices=RECIPES, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--eval-batch', type=int, default=8)
    ap.add_argument('--precision', choices=['fp16','bf16','fp32'], default='fp16')
    ap.add_argument('--smoke-steps', type=int, default=0)
    ap.add_argument('--acknowledge-duplicate-risk', action='store_true')
    a = ap.parse_args()
    if a.epochs != 100 or a.batch != 16 or a.seed not in [42,1234,31415]:
        raise ValueError('Suite is fixed: 100 epochs, batch16, seeds42/1234/31415')
    if a.eval_batch < 1 or a.smoke_steps < 0:
        raise ValueError('Invalid batch/smoke setting')
    a.repo, a.prepared, a.out = a.repo.resolve(), a.prepared.resolve(), a.out.resolve()
    if a.out.exists():
        raise FileExistsError(f'Refusing to overwrite or resume: {a.out}')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required. Use the Slurm submission script, not the login node.')
    if a.precision == 'bf16' and torch.cuda.get_device_capability()[0] < 8:
        raise RuntimeError('Native BF16 requested on an unsupported GPU')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','4')))
    folder = a.prepared / f'core6_to_{a.target}'
    protocol = json.loads((folder/'protocol.json').read_text())
    if protocol['classes'] != CLASSES6 or protocol['target'] != a.target:
        raise RuntimeError('Incorrect Core6 protocol')
    if protocol['missing_or_unreadable_images']:
        raise RuntimeError('Preparation reported unreadable images')
    if protocol['duplicate_risk'] and not (a.smoke_steps or a.acknowledge_duplicate_risk):
        raise RuntimeError('Retaining historical duplicate-risk split requires acknowledgement')
    cfg = make_config(a.recipe, 'core6', a.target, 100, 16, a.precision)
    cfg['EVAL_BATCH'] = a.eval_batch
    config, lib, original = load_project(a.repo, cfg)
    objective, objective_text = build_objective(original)
    amp = get_amp(a.precision)
    lib.set_seed(a.seed)
    device = torch.device('cuda')
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out/'checkpoints').mkdir()
    (a.out/'predictions').mkdir()
    started = time.time()
    log_file = (a.out/'train.log').open('w', encoding='utf-8', buffering=1)
    def log(text):
        text = time.strftime('%Y-%m-%dT%H:%M:%S') + ' ' + str(text)
        print(text, flush=True)
        log_file.write(text+'\n')
    packages = {}
    for name in ('torch','torchvision','timm','peft','transformers','numpy','pandas','Pillow','safetensors'):
        packages[name] = importlib.metadata.version(name)
    identity = dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
                    protocol=protocol, packages=packages, python=sys.version, gpu=torch.cuda.get_device_name(0),
                    cuda_build=torch.version.cuda, platform=platform.platform(),
                    job_id=os.environ.get('SLURM_JOB_ID'), array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'),
                    array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID'),
                    batch_id=os.environ.get('CORE6_BATCH_ID'),
                    controller='core6_target_blind_v2; pinned original objective/model/GPU augmentation; FP16 GradScaler',
                    initialization='local pretrained DINOv3 + newly initialized LoRA/heads; no old task checkpoint',
                    scientific_status='SMOKE_ONLY' if a.smoke_steps else 'TRAINING',
                    controller_sha256=sha256(Path(__file__)))
    write_json(a.out/'run_manifest.json', identity)
    write_json(a.out/'effective_config.json', serializable_config(config))
    (a.out/'audited_objective_used.py').write_text(objective_text)
    (a.out/'pip_freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    # Only source rows and source images are accessed before the final EMA is locked.
    train_rows, train_df = load_rows(folder, 'train', protocol)
    val_rows, val_df = load_rows(folder, 'val', protocol)
    source = lib.GpuSourceProvider(train_df, a.seed, device, log=log)
    val_loader = lib.build_gpu_eval_loader(val_df, device, batch=a.eval_batch)
    model = lib.TimmBackboneV921(config.DEFAULT_BACKBONE, 6, adapter_mode='lora').to(device)
    if model.weight.shape[0] != 6:
        raise RuntimeError('Classifier is not six-class')
    miro = lib.MIROProjector(config.HEAD_HIDDEN, model.enc_dim).to(device)
    adv = None
    grl_logs = grl_edges = None
    if config.V13_GRL_WEIGHT > 0:
        adv = lib.CarrierAdversary(config.HEAD_HIDDEN, hidden=config.V13_GRL_HIDDEN,
                    out_dim=3 if config.V13_GRL_DISCRETE else 1).to(device)
        grl_logs = torch.tensor([math.log(lib.parse_freq_ghz(f)) for f in config.TRAIN_FREQS],device=device)
        if config.V13_GRL_DISCRETE:
            lo,hi = lib.das_log_carrier_range()
            grl_edges = torch.linspace(lo,hi,4,device=device)[1:-1].contiguous()
    params = [p for p in model.parameters() if p.requires_grad] + list(miro.parameters())
    if adv is not None:
        params += list(adv.parameters())
    opt = torch.optim.AdamW(params, lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda', enabled=a.precision=='fp16')
    ema = lib.ModelEMA(model, config.EMA_DECAY)
    prior = lib.compute_logit_prior(train_df,6,device)
    ce = torch.nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    ce_domain = torch.nn.CrossEntropyLoss()
    write_json(a.out/'model_structure.json',dict(classes=CLASSES6,head_shape=list(model.weight.shape),
        encoder_dimension=model.enc_dim,
        lora_modules=[n for n,m in model.named_modules() if hasattr(m,'lora_A')],
        model_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
        optimizer_trainable=sum(p.numel() for p in params)))
    log(f'NEW RUN batch={identity["batch_id"]} job={identity["job_id"]} array={identity["array_job_id"]} task={identity["array_task_id"]}')
    log(f'{config.TRAIN_FREQS} -> {a.target}; Core6 {a.recipe}; seed={a.seed}; target NOT loaded')
    log(f'GPU={identity["gpu"]}; precision={a.precision}; train={len(train_rows)} val={len(val_rows)}')
    steps_per_epoch = len(source)
    total_steps, warmup_steps = 100 * steps_per_epoch, config.WARMUP_EPOCHS * steps_per_epoch
    step = updates = skipped = ema_updates = 0
    first_gradient_checked = False
    history = []
    for epoch in range(1,101):
        epoch_start = time.time()
        source.set_epoch(epoch)
        model.train(); model.set_encoder_train_mode(); miro.train()
        if adv is not None:
            adv.train()
        sums = {}; epoch_updates = epoch_skips = 0
        for _ in range(steps_per_epoch):
            lr = original.cosine_lr_schedule(step,total_steps,warmup_steps,config.LR)
            for group in opt.param_groups:
                group['lr'] = lr
            x,y,d,aux,r = next(source)
            opt.zero_grad(set_to_none=True)
            inputs = dict(model=model,miro=miro,domain_clf=None,carrier_adv=adv,
                grl_band_log_ghz=grl_logs,grl_bin_edges=grl_edges,grl_discrete=config.V13_GRL_DISCRETE,
                grl_bins=3,source_loader=source,x_src=x,y_src=y,d_src=d,x_aux=aux,r_src=r,
                global_step=step,total_steps=total_steps,logit_prior=prior,ce=ce,ce_domain=ce_domain)
            with amp():
                loss,parts,*_ = objective(**inputs)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f'Non-finite forward loss at epoch={epoch}, step={step}')
            scale_before = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            if not scaler.is_enabled() and any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
                raise FloatingPointError('Non-finite unscaled gradients')
            scaler.step(opt)
            scaler.update()
            was_skipped = scaler.get_scale() < scale_before
            if was_skipped:
                skipped += 1; epoch_skips += 1
                log(f'[AMP] skipped optimizer step={step}; loss_scale={scaler.get_scale()}')
            else:
                updates += 1; epoch_updates += 1
                if epoch >= config.EMA_START_EPOCH:
                    ema.update(model); ema_updates += 1
                # Step0 has lambda=0; require an active gradient check later.
                if not first_gradient_checked and step > 0:
                    groups = {'classification': [model.weight],
                              'encoder_adapter': [p for n,p in model.named_parameters() if n.startswith('encoder.') and p.requires_grad]}
                    if config.V13_FREQ_WEIGHT > 0:
                        groups['frequency_branch'] = list(model.freq_neck.parameters()) + list(model.freq_head.parameters())
                    if adv is not None:
                        groups['carrier_adversary'] = list(adv.parameters())
                    report = {}
                    for name,ps in groups.items():
                        gs = [p.grad for p in ps if p.grad is not None]
                        finite = bool(gs) and all(bool(torch.isfinite(g).all()) for g in gs)
                        nonzero = sum(bool((g != 0).any()) for g in gs)
                        if not finite or not nonzero:
                            raise FloatingPointError(f'Inactive/non-finite active branch: {name}')
                        report[name] = dict(gradient_tensors=len(gs), finite=finite, nonzero_tensors=nonzero)
                    write_json(a.out/'gradient_check.json',report)
                    first_gradient_checked = True
            if skipped > 100:
                raise FloatingPointError('More than 100 AMP optimizer skips; refusing a silently unstable run')
            for name,v in dict(parts,total=loss).items():
                sums[name] = sums.get(name,0.0) + float(v.detach() if torch.is_tensor(v) else v)
            step += 1
            if step % 10 == 0 or a.smoke_steps:
                log(f'[train] epoch={epoch} step={step}/{total_steps} successful_updates={updates} loss={float(loss.detach()):.5f}')
            if a.smoke_steps and updates >= a.smoke_steps and first_gradient_checked:
                # Exercise real GPU serialization/evaluation without reading target data.
                metrics,yy,before = evaluate(model,val_loader,val_rows,CLASSES6,amp)
                state = trainable_state(model)
                path = a.out/'checkpoints/smoke_live.pt'
                save_checkpoint(path,state,a,protocol,epoch,updates,skipped)
                with torch.no_grad():
                    next(p for p in model.parameters() if p.requires_grad).add_(0.125)
                load_trainable(model,torch.load(path,map_location='cpu',weights_only=True)['state'])
                _,yy2,after = evaluate(model,val_loader,val_rows,CLASSES6,amp)
                err = float(np.max(np.abs(before-after)))
                if not np.array_equal(yy,yy2) or err > 1e-5:
                    raise RuntimeError(f'Checkpoint round-trip changed predictions: {err}')
                test_ema = lib.ModelEMA(model,config.EMA_DECAY); test_ema.update(model)
                with lib.SwapEMA(model,test_ema):
                    pass
                write_json(a.out/'SMOKE_PASS.json',dict(status='PASS',successful_updates=updates,
                    target_evaluated=False,scientific_results=False,gradient_check=True,
                    checkpoint_roundtrip_max_error=err,classes=CLASSES6,
                    recipe=a.recipe,target=a.target,slurm_job_id=identity['job_id']))
                log('SMOKE_PASS: no target score, no formal training result')
                log_file.close(); return
        if not epoch_updates:
            raise FloatingPointError(f'Epoch {epoch} made no successful optimizer updates')
        val_live,_,_ = evaluate(model,val_loader,val_rows,CLASSES6,amp)
        with lib.SwapEMA(model,ema):
            val_ema,_,_ = evaluate(model,val_loader,val_rows,CLASSES6,amp)
        row = dict(epoch=epoch,attempted_steps=step,optimizer_updates=updates,
                   epoch_updates=epoch_updates,epoch_amp_skips=epoch_skips,ema_updates=ema_updates,
                   seconds=time.time()-epoch_start,losses={k:v/steps_per_epoch for k,v in sums.items()},
                   das=source.current_das_stage(),source_validation_live=val_live,
                   source_validation_ema=val_ema,target_evaluated=False)
        history.append(row); write_json(a.out/'history.json',history)
        log(f'[epoch] {epoch}/100 source_ema_F1={val_ema["macro_f1"]:.4f}; target NOT evaluated')
    if len(history) != 100 or ema_updates == 0:
        raise RuntimeError('Incomplete formal training')
    state = {k:v.detach().cpu().clone() for k,v in ema.state_dict().items()}
    path = a.out/'checkpoints/final_ema.pt'
    save_checkpoint(path,state,a,protocol,100,updates,skipped)
    write_json(a.out/'selection_frozen_before_target.json',dict(epoch=100,selection='fixed_epoch100_ema',
        target_used_for_training_or_selection=False,checkpoint_sha256=sha256(path),
        optimizer_updates=updates,ema_updates=ema_updates,amp_skipped_steps=skipped))
    # Target files are first read here, after fixed ep100 EMA is saved and hashed.
    test_rows,test_df = load_rows(folder,'test',protocol)
    loader = lib.build_gpu_eval_loader(test_df,device,batch=a.eval_batch)
    # Read back from disk and require all trainable keys, rather than silently strict=False.
    ck = torch.load(path,map_location='cpu',weights_only=True)
    load_trainable(model,ck['state'])
    metrics,yy,logits = evaluate(model,loader,test_rows,CLASSES6,amp)
    metrics = write_predictions(a.out/'predictions/final_ema',test_rows,yy,logits,CLASSES6,
        dict(seed=a.seed,target=a.target,recipe=a.recipe,epoch=100,selection='fixed_epoch100_ema'))
    write_json(a.out/'DONE.json',dict(status='COMPLETE',epoch=100,recipe=a.recipe,seed=a.seed,
        target=a.target,results={'final_ema':metrics},elapsed_seconds=time.time()-started,
        checkpoint_sha256=sha256(path),predictions_sha256=sha256(a.out/'predictions/final_ema.npz'),
        slurm_job_id=identity['job_id'],array_job_id=identity['array_job_id'],
        array_task_id=identity['array_task_id'],batch_id=identity['batch_id']))
    log(f'COMPLETE: Core6 {a.target} {a.recipe} seed{a.seed}; final_ema F1={metrics["macro_f1"]:.6f}')
    log_file.close()

if __name__ == '__main__':
    main()
