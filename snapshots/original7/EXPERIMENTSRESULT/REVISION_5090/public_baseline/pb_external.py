"""External radar-specific baselines: #7 RadMamba and #8 SelaFD.

Both are vendored under public_baseline/external/ (cloned from the official repos):
  RadMamba : https://github.com/lab-emi/AIRHAR        (Apache-2.0; pure-PyTorch SSM,
             NO mamba-ssm/causal-conv1d CUDA kernels -> runs on Win+sm_120+torch2.11)
  SelaFD   : https://github.com/wangyijunlyy/SelaFD   (NO LICENSE file in repo; ViT-B/16
             + LoRA(weight) + serial/parallel Adapter(feature); pure PyTorch)

Design rule (PUBLIC_BASELINE_DESIGN.md S5): radar-specific methods keep their NATIVE
architecture + optimizer/loss, but to stay comparable we overlay the SHARED protocol:
  - our 7-class 10/24->77 data, the SAME 224x224x3 per-image-standardized input (pb_lib cache);
  - 100 epochs, class-balanced sampler, final-EMA checkpoint, full-418 + source-val eval,
    NO target / source-val checkpoint selection.
Native bits kept per method: RadMamba = AdamW lr5e-3 + grad-clip 200, CE; SelaFD = Adam
lr1e-3 + CosineAnnealing, CE(label-smoothing 0.1), LoRA r4/alpha4 + adapters on a frozen
ImageNet ViT-B/16.

Modes:
  --probe            build both models (num_classes=7), forward ONE real batch, print shapes/params
  --smoke KEY        1-epoch run of one method to a temp dir (pipeline sanity)
  --train KEY --seed S --run-dir D    full 100ep final-EMA run (KEY in {radmamba, selafd})
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

import pb_lib as P

HERE = Path(__file__).resolve().parent
EXT = HERE / "external"
AIRHAR = EXT / "AIRHAR"
SELAFD = EXT / "SelaFD"
SELAFD_PRETRAIN = SELAFD / "pretrain" / "vit_base_patch16_224.pth"


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------
def build_radmamba(num_classes=7, channels=3, dim=80, d_state=4, depth=1,
                   image_size=224, cc_layer=1, cc_out=3, td_factor=2):
    """RadMamba backbone (arguments.py defaults: dim80, d_state4, depth1, cc1/3, td2).
    Includes its own mean-pool + LayerNorm + Linear output head -> returns (B, num_classes)."""
    if str(AIRHAR) not in sys.path:
        sys.path.insert(0, str(AIRHAR))           # so `backbones.RadMamba`/`backbones.SSM` resolve
    from backbones.RadMamba import RadMamba
    model = RadMamba(
        dim=dim, dt_rank=0, dim_inner=dim, d_state=d_state, num_classes=num_classes,
        image_height=image_size, image_width=image_size, channels=channels,
        dropout=0.0, depth=depth, channel_confusion_layer=cc_layer,
        channel_confusion_out_channels=cc_out, time_downsample_factor=td_factor,
        optional_avg_pool=False,
    )
    return model


def _ensure_selafd_pretrained(log=print):
    """Filtered timm vit_base_patch16_224 (ImageNet) state_dict matching SelaFD's vendored
    VisionTransformer keys -> saved as the .pth get_backbone() expects. Built once."""
    if SELAFD_PRETRAIN.exists():
        return SELAFD_PRETRAIN
    if str(SELAFD) not in sys.path:
        sys.path.insert(0, str(SELAFD))
    import timm
    from vit_timm import vit_base_patch16_224_fix
    backbone = vit_base_patch16_224_fix()                    # pre-surgery; same keys as timm ViT-B
    bk = backbone.state_dict()
    src = timm.create_model("vit_base_patch16_224", pretrained=True).state_dict()
    keep = {k: v for k, v in src.items() if k in bk and bk[k].shape == v.shape}
    SELAFD_PRETRAIN.parent.mkdir(parents=True, exist_ok=True)
    torch.save(keep, SELAFD_PRETRAIN)
    log(f"[selafd] built pretrained .pth: {len(keep)}/{len(bk)} backbone keys matched -> {SELAFD_PRETRAIN.name}")
    return SELAFD_PRETRAIN


def build_selafd(num_classes=7, rank=4, alpha=4, log=print):
    if str(SELAFD) not in sys.path:
        sys.path.insert(0, str(SELAFD))
    _ensure_selafd_pretrained(log=log)
    from lora import LoRA_ViT_timm
    model = LoRA_ViT_timm(r=rank, alpha=alpha, num_classes=num_classes,
                          pretrained_path=str(SELAFD_PRETRAIN))
    return model


BUILDERS = {
    "radmamba": lambda nc, log=print: build_radmamba(num_classes=nc),
    "selafd":   lambda nc, log=print: build_selafd(num_classes=nc, log=log),
}
# native recipe per method: (optimizer_name, lr, weight_decay, grad_clip, only_trainable)
NATIVE = {
    "radmamba": dict(opt="adamw", lr=5e-3, wd=0.0,  grad_clip=200.0, only_trainable=False),
    "selafd":   dict(opt="adam",  lr=1e-3, wd=0.0,  grad_clip=0.0,   only_trainable=True),
}

# Strong-but-fair external recipes: native architectures are kept; the changes are
# stabilization only and are fixed before scoring (no target/source-val selection).
STRONG = {
    "radmamba": dict(opt="adamw", lr=2e-3, wd=1e-4, grad_clip=50.0, only_trainable=False,
                     label_smoothing=0.00, ema_decay=0.995,
                     aug_intensity=0.10, aug_noise=0.015, aug_mask_prob=0.50, aug_mask_frac=0.08,
                     model_cfg={}),
    "selafd":   dict(opt="adam",  lr=5e-4, wd=0.0,  grad_clip=1.0,  only_trainable=True,
                     label_smoothing=0.05, ema_decay=0.995,
                     aug_intensity=0.08, aug_noise=0.010, aug_mask_prob=0.40, aug_mask_frac=0.06,
                     model_cfg={"rank": 8, "alpha": 8}),
}


def build_model(key, num_classes=7, model_cfg=None, log=print):
    model_cfg = dict(model_cfg or {})
    if key == "radmamba":
        return build_radmamba(num_classes=num_classes, **model_cfg)
    if key == "selafd":
        return build_selafd(num_classes=num_classes,
                            rank=int(model_cfg.get("rank", 4)),
                            alpha=int(model_cfg.get("alpha", 4)),
                            log=log)
    raise KeyError(key)


def recipe_config(key, recipe):
    cfg = dict(NATIVE[key])
    cfg.update(dict(label_smoothing=0.05 if key == "radmamba" else 0.1,
                    ema_decay=0.999,
                    aug_intensity=0.0, aug_noise=0.0, aug_mask_prob=0.0,
                    aug_mask_frac=0.0, model_cfg={}))
    if recipe == "strong":
        cfg.update(STRONG[key])
    return cfg


# --------------------------------------------------------------------------
# Probe: instantiate both + forward one real batch
# --------------------------------------------------------------------------
def probe():
    _train, _val, test_df = P.load_manifests()
    batches = P.build_eval_batches(test_df, batch=8)
    x, y, _d = batches[0]
    print(f"[probe] input batch x={tuple(x.shape)} (per-image standardized, our 224x224x3)")
    ok = True
    for key in ("radmamba", "selafd"):
        try:
            m = BUILDERS[key](7).to(P.DEVICE).eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=P.AMP):
                out = m(x)
            ntr = sum(p.numel() for p in m.parameters() if p.requires_grad)
            ntot = sum(p.numel() for p in m.parameters())
            assert out.shape == (x.shape[0], 7), f"bad output shape {tuple(out.shape)}"
            print(f"  OK  {key:9s} out={tuple(out.shape)} trainable={ntr/1e6:.3f}M total={ntot/1e6:.2f}M")
            del m; torch.cuda.empty_cache()
        except Exception as e:
            ok = False
            import traceback
            print(f"  FAIL {key}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print("EXTERNAL PROBE OK" if ok else "EXTERNAL PROBE FAILED")
    return ok


# --------------------------------------------------------------------------
# Trainer (native optimizer + shared protocol via pb_lib)
# --------------------------------------------------------------------------
def make_optimizer(model, rc):
    params = [p for p in model.parameters() if p.requires_grad] if rc["only_trainable"] else list(model.parameters())
    if rc["opt"] == "adamw":
        return torch.optim.AdamW(params, lr=rc["lr"], weight_decay=rc["wd"]), rc
    return torch.optim.Adam(params, lr=rc["lr"], weight_decay=rc["wd"]), rc


def apply_noncarrier_aug(x, rc):
    if rc.get("aug_mask_prob", 0.0) <= 0 and rc.get("aug_intensity", 0.0) <= 0 and rc.get("aug_noise", 0.0) <= 0:
        return x
    x = x.clone()
    b, _c, h, w = x.shape
    a = float(rc.get("aug_intensity", 0.0))
    if a > 0:
        mean = x.mean(dim=(2, 3), keepdim=True)
        contrast = torch.empty(b, 1, 1, 1, device=x.device).uniform_(1.0 - a, 1.0 + a)
        brightness = torch.empty(b, 1, 1, 1, device=x.device).uniform_(-0.5 * a, 0.5 * a)
        x = (x - mean) * contrast + mean + brightness
    noise = float(rc.get("aug_noise", 0.0))
    if noise > 0:
        sigma = torch.empty(b, 1, 1, 1, device=x.device).uniform_(0.0, noise)
        x = x + torch.randn_like(x) * sigma
    x = x.clamp(0.0, 1.0)
    p = float(rc.get("aug_mask_prob", 0.0))
    frac = float(rc.get("aug_mask_frac", 0.0))
    if p > 0 and frac > 0:
        max_w = max(1, int(round(w * frac)))
        max_h = max(1, int(round(h * frac)))
        for i in range(b):
            if torch.rand((), device=x.device) >= p:
                continue
            fill = x[i].mean(dim=(1, 2), keepdim=True)
            tw = int(torch.randint(1, max_w + 1, (), device=x.device).item())
            t0 = int(torch.randint(0, max(1, w - tw + 1), (), device=x.device).item())
            x[i, :, :, t0:t0 + tw] = fill
            fh = int(torch.randint(1, max_h + 1, (), device=x.device).item())
            f0 = int(torch.randint(0, max(1, h - fh + 1), (), device=x.device).item())
            x[i, :, f0:f0 + fh, :] = fill
    return x


def train_one(key, seed, run_dir, epochs=100, recipe="native", ema_start=5,
              warmup_epochs=3, batch_size=16, label_smoothing=0.05, eval_every=5):
    P.set_seed(seed)
    run_dir = Path(run_dir)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)
    log_lines = []

    def log(m):
        line = f"[{datetime.now():%H:%M:%S}] {m}"
        print(line, flush=True); log_lines.append(line)

    rc = recipe_config(key, recipe)
    if label_smoothing != 0.05:
        rc["label_smoothing"] = label_smoothing
    log(f"method={key} seed={seed} recipe={recipe} device={P.DEVICE} run_dir={run_dir}")
    train_df, val_df, test_df = P.load_manifests()
    log(f"splits: train={len(train_df)} val(source)={len(val_df)} test77={len(test_df)}")
    imgs, ys, ds, fs = P.build_train_cache(train_df)
    val_batches = P.build_eval_batches(val_df)
    test_batches = P.build_eval_batches(test_df)
    weights = P.class_balanced_weights(train_df, P.DEVICE)
    gen = torch.Generator(device=P.DEVICE); gen.manual_seed(int(seed))
    std = P.lib.gpu_standardize
    n = imgs.shape[0]
    spe = max(1, n // batch_size)
    total_steps = epochs * spe
    warmup_steps = warmup_epochs * spe

    model_cfg = dict(rc.get("model_cfg", {}))
    model = build_model(key, 7, model_cfg=model_cfg, log=log).to(P.DEVICE)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_to = sum(p.numel() for p in model.parameters())
    log(f"trainable={n_tr/1e6:.3f}M / total={n_to/1e6:.2f}M")
    opt, rc = make_optimizer(model, rc)
    ce = nn.CrossEntropyLoss(label_smoothing=float(rc["label_smoothing"]))
    ema = P.ModelEMA(model, float(rc["ema_decay"]))
    log(f"native: opt={rc['opt']} lr={rc['lr']} wd={rc['wd']} grad_clip={rc['grad_clip']} "
        f"ema_decay={rc['ema_decay']} ema_start={ema_start} epochs={epochs} steps/epoch={spe} "
        f"ce_ls={rc['label_smoothing']} model_cfg={model_cfg} "
        f"aug=(intensity={rc['aug_intensity']},noise={rc['aug_noise']},"
        f"mask_p={rc['aug_mask_prob']},mask_frac={rc['aug_mask_frac']})")
    last_eval = {}

    def write_progress(ep, step_in_epoch=0, run_loss=None, status="training"):
        payload = {
            "status": status,
            "epoch": int(ep),
            "total": int(epochs),
            "step_in_epoch": int(step_in_epoch),
            "steps_per_epoch": int(spe),
            "global_step": int(gstep),
            "total_steps": int(total_steps),
            "name": key,
            "seed": int(seed),
            "family": "external",
            "recipe": recipe,
            "train_loss": (float(run_loss) if run_loss is not None else None),
            "lr": float(opt.param_groups[0]["lr"]),
        }
        payload.update(last_eval)
        (run_dir / "progress.json").write_text(json.dumps(payload), encoding="utf-8")

    def eval_ema(batches):
        backup = {k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}
        ema.copy_to(model); model.eval()
        out = P.eval_forward(lambda z: model(z), batches)
        sd = model.state_dict()
        for k2, v2 in backup.items():
            sd[k2].copy_(v2)
        return out

    history = []
    gstep = 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        run_loss = 0.0
        for step_in_epoch in range(1, spe + 1):
            idx = torch.multinomial(weights, batch_size, replacement=True, generator=gen)
            x = std(apply_noncarrier_aug(imgs[idx], rc)); yb = ys[idx]
            lr_now = P.cosine_lr(gstep, total_steps, warmup_steps, rc["lr"])
            for g in opt.param_groups:
                g["lr"] = lr_now
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=P.AMP):
                logits = model(x)
                loss = ce(logits, yb)
            loss.backward()
            if rc["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), rc["grad_clip"])
            opt.step()
            run_loss += float(loss.detach()); gstep += 1
            if ep >= ema_start:
                ema.update(model)
            if step_in_epoch == 1 or step_in_epoch % 10 == 0 or step_in_epoch == spe:
                write_progress(ep, step_in_epoch, run_loss / step_in_epoch)
        do_eval = (ep == 1 or ep == epochs or ep % eval_every == 0)
        if ep >= ema_start and do_eval:
            v = eval_ema(val_batches); t = eval_ema(test_batches)
            last_eval = {
                "last_eval_epoch": int(ep),
                "val_f1_ema": float(v["macro_f1"]),
                "test77_acc_ema": float(t["acc"]),
                "test77_f1_ema": float(t["macro_f1"]),
                "gen_gap": float(v["macro_f1"] - t["macro_f1"]),
            }
            history.append({"epoch": ep, "train_loss": run_loss / spe,
                            "val_f1_ema": v["macro_f1"], "test77_f1_ema": t["macro_f1"],
                            "test77_acc_ema": t["acc"]})
            log(f"ep {ep:03d}/{epochs} loss={run_loss/spe:.4f} val_ema_f1={v['macro_f1']:.4f} "
                f"test77_ema={t['acc']:.4f}/{t['macro_f1']:.4f} lr={lr_now:.2e}")
        else:
            history.append({"epoch": ep, "train_loss": run_loss / spe})
        write_progress(ep, spe, run_loss / spe)

    train_time = time.time() - t0
    ckpt = {"ema": ema.state_dict(), "epoch": epochs, "method": key, "family": "external",
            "seed": seed, "n_trainable": int(n_tr), "n_total": int(n_to),
            "recipe": recipe, "model_cfg": model_cfg,
            "train_cfg": {k: v for k, v in rc.items() if k != "model_cfg"}}
    ckpt_path = run_dir / "checkpoints" / f"pool_ep{epochs}_ema.pt"
    torch.save(ckpt, ckpt_path)
    ft = eval_ema(test_batches); fv = eval_ema(val_batches)
    summary = {"method": key, "family": "external", "seed": seed, "epochs": epochs,
               "recipe": recipe, "model_cfg": model_cfg,
               "n_trainable": int(n_tr), "n_total": int(n_to), "train_time_s": train_time,
               "final_ema": {"test77_acc": ft["acc"], "test77_macro_f1": ft["macro_f1"],
                             "test77_per_class_f1": ft["per_class_f1"],
                             "source_val_macro_f1": fv["macro_f1"],
                             "gen_gap": fv["macro_f1"] - ft["macro_f1"]}}
    (run_dir / "reports" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "reports" / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (run_dir / "train.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    log(f"DONE {key} seed{seed}: final-EMA test77={ft['acc']:.4f}/{ft['macro_f1']:.4f} "
        f"src-val f1={fv['macro_f1']:.4f} gap={summary['final_ema']['gen_gap']:+.4f} ({train_time/60:.1f} min)")
    write_progress(epochs, spe, history[-1]["train_loss"], status="done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--smoke", default=None, choices=list(BUILDERS))
    ap.add_argument("--train", default=None, choices=list(BUILDERS))
    ap.add_argument("--recipe", choices=["native", "strong"], default="native")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()

    if args.probe:
        sys.exit(0 if probe() else 1)
    if args.smoke:
        rd = HERE / "runs" / "_smoke_ext" / args.smoke
        train_one(args.smoke, seed=42, run_dir=rd, epochs=1, recipe=args.recipe, ema_start=1, eval_every=1)
        sys.exit(0)
    if args.train:
        rd = args.run_dir or (HERE / "runs" / args.train / f"seed{args.seed}")
        train_one(args.train, seed=args.seed, run_dir=rd, epochs=args.epochs, recipe=args.recipe)
        sys.exit(0)
    ap.error("choose --probe / --smoke KEY / --train KEY")


if __name__ == "__main__":
    main()
