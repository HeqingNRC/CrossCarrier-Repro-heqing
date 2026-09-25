"""Train ONE generic image backbone (#1-6) under the unified public-baseline protocol.

Full fine-tune, 100 epochs, final-EMA checkpoint, per-image-standardized input,
class-balanced sampling, plain CE + label smoothing 0.05. NO DAS / HFT / SpecAugment
(those are the proposed method's contribution -- the whole point is the gap).

Writes, under <run-dir>/:
  checkpoints/pool_ep100_ema.pt   {"ema": <shadow state_dict>, "epoch":100, ...}
  reports/history.json            per-epoch val/test (EMA) macro-F1 for audit
  reports/summary.json            final-EMA metrics + trainable params + per-class
  train.log

Eval here is for the HISTORY/audit trail only; the AUTHORITATIVE numbers come from
pb_eval_unified.py which re-loads pool_ep100_ema.pt and evaluates every method with
one identical metric path. (They will match -- same forward, same metric.)

Usage:
  python pb_train_generic.py --backbone convnext_t --seed 42 \
      --run-dir EXPERIMENTSRESULT/REVISION_5090/public_baseline/runs/convnext_t/seed42
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

import pb_lib as P

BASE_DEFAULTS = dict(
    lr=3e-4,
    backbone_lr=None,
    head_lr=None,
    weight_decay=0.05,
    batch_size=16,
    warmup_epochs=3,
    label_smoothing=0.05,
    ema_decay=0.999,
    ema_start=5,
    dropout=0.2,
    head_hidden=0,
    freeze_backbone_epochs=0,
    grad_clip=0.0,
    mixup_alpha=0.0,
    aug_intensity=0.0,
    aug_noise=0.0,
    aug_mask_prob=0.0,
    aug_mask_frac=0.0,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=list(P.GENERIC_BACKBONES))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--recipe", choices=["plain", "strong"], default="plain",
                    help="plain preserves the original public-baseline recipe; strong applies "
                         "pre-registered source-only stabilization, still final-EMA/no target.")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--backbone-lr", type=float, default=None)
    ap.add_argument("--head-lr", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--warmup-epochs", type=int, default=None)
    ap.add_argument("--label-smoothing", type=float, default=None)
    ap.add_argument("--ema-decay", type=float, default=None)
    ap.add_argument("--ema-start", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--head-hidden", type=int, default=None)
    ap.add_argument("--freeze-backbone-epochs", type=int, default=None)
    ap.add_argument("--grad-clip", type=float, default=None)
    ap.add_argument("--mixup-alpha", type=float, default=None)
    ap.add_argument("--aug-intensity", type=float, default=None,
                    help="raw-image brightness/contrast jitter strength; no flips/no carrier stretch.")
    ap.add_argument("--aug-noise", type=float, default=None,
                    help="max Gaussian noise std on raw [0,1] images before standardization.")
    ap.add_argument("--aug-mask-prob", type=float, default=None,
                    help="probability of non-carrier time/Doppler masking per sample.")
    ap.add_argument("--aug-mask-frac", type=float, default=None,
                    help="max width/height fraction for non-carrier masks.")
    ap.add_argument("--eval-every", type=int, default=5,
                    help="diagnostic history eval cadence; ep1 and last always eval. "
                         "Does NOT affect the final-EMA checkpoint that gets scored.")
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    return finalize_args(args)


def finalize_args(args):
    cfg = dict(BASE_DEFAULTS)
    if args.recipe == "strong":
        cfg.update(P.GENERIC_STRONG_RECIPES[args.backbone])
    for key in BASE_DEFAULTS:
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val
    for key, val in cfg.items():
        setattr(args, key, val)
    if args.backbone_lr is None:
        args.backbone_lr = args.lr
    if args.head_lr is None:
        args.head_lr = args.lr
    return args


def apply_noncarrier_aug(x, args):
    """Standard source-only image robustness aug: no flips, no DAS, no carrier ratios."""
    if args.recipe != "strong":
        return x
    x = x.clone()
    b, _c, h, w = x.shape
    if args.aug_intensity and args.aug_intensity > 0:
        a = float(args.aug_intensity)
        mean = x.mean(dim=(2, 3), keepdim=True)
        contrast = torch.empty(b, 1, 1, 1, device=x.device).uniform_(1.0 - a, 1.0 + a)
        brightness = torch.empty(b, 1, 1, 1, device=x.device).uniform_(-0.5 * a, 0.5 * a)
        x = (x - mean) * contrast + mean + brightness
    if args.aug_noise and args.aug_noise > 0:
        sigma = torch.empty(b, 1, 1, 1, device=x.device).uniform_(0.0, float(args.aug_noise))
        x = x + torch.randn_like(x) * sigma
    x = x.clamp(0.0, 1.0)
    if args.aug_mask_prob and args.aug_mask_frac and args.aug_mask_prob > 0 and args.aug_mask_frac > 0:
        max_w = max(1, int(round(w * float(args.aug_mask_frac))))
        max_h = max(1, int(round(h * float(args.aug_mask_frac))))
        for i in range(b):
            if torch.rand((), device=x.device) >= float(args.aug_mask_prob):
                continue
            fill = x[i].mean(dim=(1, 2), keepdim=True)
            tw = int(torch.randint(1, max_w + 1, (), device=x.device).item())
            t0 = int(torch.randint(0, max(1, w - tw + 1), (), device=x.device).item())
            x[i, :, :, t0:t0 + tw] = fill
            fh = int(torch.randint(1, max_h + 1, (), device=x.device).item())
            f0 = int(torch.randint(0, max(1, h - fh + 1), (), device=x.device).item())
            x[i, :, f0:f0 + fh, :] = fill
    return x


def apply_mixup(x, y, alpha):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    perm = torch.randperm(y.numel(), device=y.device)
    return lam * x + (1.0 - lam) * x[perm], y, y[perm], lam


def main():
    args = parse_args()
    P.set_seed(args.seed)
    run_dir = Path(args.run_dir)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)

    log_lines = []

    def log(msg):
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    log(f"backbone={args.backbone} ({P.GENERIC_BACKBONES[args.backbone]}) seed={args.seed} "
        f"recipe={args.recipe} device={P.DEVICE} run_dir={run_dir}")

    # ---- data ----
    train_df, val_df, test_df = P.load_manifests()
    log(f"splits: train={len(train_df)} val(source)={len(val_df)} test77={len(test_df)}")
    imgs, ys, ds, fs = P.build_train_cache(train_df)        # raw [0,1] in VRAM
    val_batches = P.build_eval_batches(val_df)              # standardized once
    test_batches = P.build_eval_batches(test_df)
    weights = P.class_balanced_weights(train_df, P.DEVICE)
    gen = torch.Generator(device=P.DEVICE); gen.manual_seed(int(args.seed))
    n = imgs.shape[0]
    steps_per_epoch = max(1, n // args.batch_size)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    # ---- model / opt / ema ----
    model = P.GenericBackbone(
        args.backbone, dropout=args.dropout, head_hidden=args.head_hidden).to(P.DEVICE)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    log(f"trainable={n_trainable/1e6:.2f}M / total={n_total/1e6:.2f}M  "
        f"feat_dim={model.feat_dim} head_hidden={args.head_hidden} dropout={args.dropout}")
    opt = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": args.backbone_lr,
             "_base_lr": args.backbone_lr, "_name": "backbone"},
            {"params": model.head.parameters(), "lr": args.head_lr,
             "_base_lr": args.head_lr, "_name": "head"},
        ],
        weight_decay=args.weight_decay,
    )
    ce = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    ema = P.ModelEMA(model, args.ema_decay)
    log(f"epochs={args.epochs} steps/epoch={steps_per_epoch} total_steps={total_steps} "
        f"lr={args.lr} backbone_lr={args.backbone_lr} head_lr={args.head_lr} "
        f"wd={args.weight_decay} batch={args.batch_size} warmup_ep={args.warmup_epochs} "
        f"ema_decay={args.ema_decay} ema_start={args.ema_start} "
        f"freeze_backbone_ep={args.freeze_backbone_epochs} grad_clip={args.grad_clip} "
        f"mixup={args.mixup_alpha} aug=(intensity={args.aug_intensity},noise={args.aug_noise},"
        f"mask_p={args.aug_mask_prob},mask_frac={args.aug_mask_frac})")

    # local handle to GPU standardize (per-image; == PerImageStandardize)
    lib_standardize = P.lib.gpu_standardize
    last_eval = {}

    def write_progress(ep, step_in_epoch=0, run_loss=None, status="training"):
        payload = {
            "status": status,
            "epoch": int(ep),
            "total": int(args.epochs),
            "step_in_epoch": int(step_in_epoch),
            "steps_per_epoch": int(steps_per_epoch),
            "global_step": int(gstep),
            "total_steps": int(total_steps),
            "name": args.backbone,
            "seed": int(args.seed),
            "family": "generic",
            "recipe": args.recipe,
            "train_loss": (float(run_loss) if run_loss is not None else None),
            "lr_backbone": float(opt.param_groups[0]["lr"]),
            "lr_head": float(opt.param_groups[1]["lr"]),
        }
        payload.update(last_eval)
        (run_dir / "progress.json").write_text(json.dumps(payload), encoding="utf-8")

    def eval_ema(batches):
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()
                  if v.dtype.is_floating_point}
        ema.copy_to(model); model.eval()
        out = P.eval_forward(lambda x: model(x), batches)   # batches already standardized
        sd = model.state_dict()
        for k, v in backup.items():
            sd[k].copy_(v)
        return out

    history = []
    gstep = 0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        for step_in_epoch in range(1, steps_per_epoch + 1):
            idx = torch.multinomial(weights, args.batch_size, replacement=True, generator=gen)
            x = lib_standardize(apply_noncarrier_aug(imgs[idx], args)); y = ys[idx]
            x, y_a, y_b, mix_lam = apply_mixup(x, y, float(args.mixup_alpha))
            for g in opt.param_groups:
                lr_now = P.cosine_lr(gstep, total_steps, warmup_steps, g["_base_lr"])
                if g["_name"] == "backbone" and ep <= int(args.freeze_backbone_epochs):
                    lr_now = 0.0
                g["lr"] = lr_now
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=P.AMP):
                logits = model(x)
                loss = mix_lam * ce(logits, y_a) + (1.0 - mix_lam) * ce(logits, y_b)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            opt.step()
            run_loss += float(loss.detach())
            gstep += 1
            if ep >= args.ema_start:
                ema.update(model)
            if step_in_epoch == 1 or step_in_epoch % 10 == 0 or step_in_epoch == steps_per_epoch:
                write_progress(ep, step_in_epoch, run_loss / step_in_epoch)

        do_eval = (ep == 1 or ep == args.epochs or ep % args.eval_every == 0)
        if ep >= args.ema_start and do_eval:
            v = eval_ema(val_batches); t = eval_ema(test_batches)
            last_eval = {
                "last_eval_epoch": int(ep),
                "val_acc_ema": float(v["acc"]),
                "val_f1_ema": float(v["macro_f1"]),
                "test77_acc_ema": float(t["acc"]),
                "test77_f1_ema": float(t["macro_f1"]),
                "gen_gap": float(v["macro_f1"] - t["macro_f1"]),
            }
            history.append({"epoch": ep, "train_loss": run_loss / steps_per_epoch,
                            "val_f1_ema": v["macro_f1"], "val_acc_ema": v["acc"],
                            "test77_f1_ema": t["macro_f1"], "test77_acc_ema": t["acc"]})
            log(f"ep {ep:03d}/{args.epochs} loss={run_loss/steps_per_epoch:.4f} "
                f"val_ema={v['acc']:.4f}/{v['macro_f1']:.4f} "
                f"test77_ema={t['acc']:.4f}/{t['macro_f1']:.4f} "
                f"lr_bb={opt.param_groups[0]['lr']:.2e} lr_head={opt.param_groups[1]['lr']:.2e}")
        else:
            history.append({"epoch": ep, "train_loss": run_loss / steps_per_epoch})
            if ep % 10 == 0:
                log(f"ep {ep:03d}/{args.epochs} loss={run_loss/steps_per_epoch:.4f} (no eval)")
        write_progress(ep, steps_per_epoch, run_loss / steps_per_epoch)

    train_time = time.time() - t0

    # ---- final-EMA checkpoint (the authoritative selection) ----
    ckpt = {
        "ema": ema.state_dict(), "epoch": args.epochs,
        "backbone": args.backbone, "timm_name": model.timm_name,
        "family": "generic", "seed": args.seed,
        "n_trainable": int(n_trainable), "n_total": int(n_total),
        "selection": "final_ema_ep%d" % args.epochs,
        "recipe": args.recipe,
        "model_cfg": {"dropout": float(args.dropout), "head_hidden": int(args.head_hidden)},
        "train_cfg": {
            "lr": float(args.lr), "backbone_lr": float(args.backbone_lr),
            "head_lr": float(args.head_lr), "weight_decay": float(args.weight_decay),
            "batch_size": int(args.batch_size), "warmup_epochs": int(args.warmup_epochs),
            "label_smoothing": float(args.label_smoothing), "ema_decay": float(args.ema_decay),
            "ema_start": int(args.ema_start),
            "freeze_backbone_epochs": int(args.freeze_backbone_epochs),
            "grad_clip": float(args.grad_clip), "mixup_alpha": float(args.mixup_alpha),
            "aug_intensity": float(args.aug_intensity), "aug_noise": float(args.aug_noise),
            "aug_mask_prob": float(args.aug_mask_prob), "aug_mask_frac": float(args.aug_mask_frac),
        },
    }
    ckpt_path = run_dir / "checkpoints" / f"pool_ep{args.epochs}_ema.pt"
    torch.save(ckpt, ckpt_path)

    final_val = eval_ema(val_batches)
    final_test = eval_ema(test_batches)
    summary = {
        "backbone": args.backbone, "timm_name": model.timm_name, "family": "generic",
        "seed": args.seed, "epochs": args.epochs, "recipe": args.recipe,
        "n_trainable": int(n_trainable), "n_total": int(n_total),
        "model_cfg": {"dropout": float(args.dropout), "head_hidden": int(args.head_hidden)},
        "train_cfg": ckpt["train_cfg"],
        "train_time_s": train_time,
        "final_ema": {
            "test77_acc": final_test["acc"], "test77_macro_f1": final_test["macro_f1"],
            "test77_per_class_f1": final_test["per_class_f1"],
            "source_val_acc": final_val["acc"], "source_val_macro_f1": final_val["macro_f1"],
            "gen_gap": final_val["macro_f1"] - final_test["macro_f1"],
        },
        "classes": list(P.config.CLASSES),
    }
    (run_dir / "reports" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "reports" / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (run_dir / "train.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    log(f"DONE {args.backbone} seed{args.seed}: final-EMA test77 acc/f1="
        f"{final_test['acc']:.4f}/{final_test['macro_f1']:.4f} "
        f"src-val f1={final_val['macro_f1']:.4f} gap={summary['final_ema']['gen_gap']:+.4f} "
        f"({train_time/60:.1f} min)  saved {ckpt_path.name}")
    write_progress(args.epochs, steps_per_epoch, history[-1]["train_loss"], status="done")


if __name__ == "__main__":
    main()
