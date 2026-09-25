"""Strict-only training loop shared by V15-V18.

The version-specific method lives in each baseline_v*/config.py and
baseline_v*/v9_2_1lib.py. This file only centralizes the protocol:
source-val EMA checkpoint selection, diagnostic 77GHz final-test monitoring,
and strict checkpoint naming.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys as _sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))
import pool_protocol  # noqa: E402

import config
from v9_2_1lib import (
    CarrierAdversary,
    DomainClassifier,
    GpuSourceProvider,
    MIROProjector,
    ModelEMA,
    SwapEMA,
    TimmBackboneV921,
    build_gpu_eval_loader,
    compute_logit_prior,
    das_stage_for_epoch,
    evaluate,
    grad_reverse,
    load_manifest,
    load_or_build_test77_dev_final,
    make_eval_loader,
    make_source_train_loader,
    set_seed,
    supcon_loss,
    write_v9_2_1_manifests,
)


def parse_args():
    ap = argparse.ArgumentParser(description=config.EXPERIMENT_NAME)
    ap.add_argument("--stage", choices=["train", "eval"], default="train")
    ap.add_argument("--backbone", default=config.DEFAULT_BACKBONE)
    ap.add_argument(
        "--adapter-mode",
        choices=["lora", "last_block", "full_ft", "frozen"],
        default=config.BACKBONE_TUNE_MODE,
    )
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument(
        "--progress",
        choices=["simple", "detailed"],
        default=config.TERMINAL_PROGRESS,
    )
    ap.add_argument(
        "--ckpt",
        choices=["sourceval", "sourcequalified", "ema", "best", "last"],
        default="sourceval",
    )
    ap.add_argument("--seed", type=int, default=config.SEED)
    # Resume a previous run to train MORE epochs without retraining. Point
    # --run-dir at the SAME dir so epoch_ckpts/ accumulates, set --epochs to the
    # NEW total (e.g. 150), and --resume <run_dir>/resume.pt. The cosine LR is
    # re-targeted to the new horizon, so resumed epochs run at a small decaying
    # LR (warm continuation). Requires the source run trained with SAVE_RESUME=1.
    ap.add_argument("--resume", default=None)
    return ap.parse_args()


def amp_dtype():
    return torch.bfloat16 if config.AMP_DTYPE == "bf16" else torch.float16


def autocast_device_type(device):
    return "cuda" if device.type == "cuda" else "cpu"


def safe_name(backbone):
    return backbone.replace("/", "_").replace(".", "_").replace(":", "_")


def atomic_save(obj, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def save_resume_bundle(path, *, ep, global_step, steps_per_epoch, warmup_steps,
                       model, ema, miro, domain_clf, opt, args):
    """Full state needed to CONTINUE training later (optimizer + EMA + MIRO +
    RNG + counters). Trainable-only model weights (frozen backbone is rebuilt
    from pretrained on resume), so the bundle stays small (~tens of MB)."""
    import random as _random
    import numpy as _np
    tnames = set(model.trainable_parameter_names())
    bundle = {
        "format": "v20_resume_v1",
        "epoch": int(ep),                       # last COMPLETED epoch
        "global_step": int(global_step),
        "steps_per_epoch": int(steps_per_epoch),
        "warmup_steps": int(warmup_steps),
        "base_lr": float(config.LR),
        "trained_epochs": int(args.epochs),      # horizon this bundle was made under
        "model_trainable": {
            k: v.detach().to("cpu", torch.float32).clone()
            for k, v in model.state_dict().items() if k in tnames
        },
        "ema": {k: v.detach().to("cpu").clone() for k, v in ema.state_dict().items()},
        "miro": {k: v.detach().to("cpu").clone() for k, v in miro.state_dict().items()},
        "domain_clf": (
            {k: v.detach().to("cpu").clone() for k, v in domain_clf.state_dict().items()}
            if domain_clf is not None else None
        ),
        "opt": opt.state_dict(),
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
            "numpy": _np.random.get_state(),
            "python": _random.getstate(),
        },
        "backbone": args.backbone,
        "adapter_mode": args.adapter_mode,
        "experiment_name": config.EXPERIMENT_NAME,
    }
    atomic_save(bundle, path)


def gpu_mem():
    if not torch.cuda.is_available():
        return "cpu"
    used = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    return f"{used:.1f}/{reserved:.1f}GB"


def current_lr(opt):
    return float(opt.param_groups[0]["lr"]) if opt.param_groups else 0.0


def detailed_progress():
    return config.TERMINAL_PROGRESS == "detailed"


def terminal_summary_message(msg):
    if msg.startswith(("run_dir=", "device=", "backbone=", "env:")):
        return True
    keys = (
        " train_source=", "heldout77_unused=", " done ", "77GHz ",
        " eval ", "freq_scan", "ep_summary", "strict77_monitor",
        "sourcequalified", "best_sourceval_ema", "saved ",
    )
    return any(k in msg for k in keys)


def make_run_dir(args):
    if args.run_dir:
        rd = Path(args.run_dir)
    else:
        tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        rd = config.RUNS_DIR / f"{tag}_{config.EXPERIMENT_NAME}"
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "checkpoints").mkdir(exist_ok=True)
    (rd / "reports").mkdir(exist_ok=True)
    return rd


def log_factory(run_dir):
    f = open(run_dir / "train.log", "a", encoding="utf-8")

    def log(msg):
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        if detailed_progress() or terminal_summary_message(msg):
            tqdm.write(line)
        f.write(line + "\n")
        f.flush()

    return log, f


def build_dataloaders(log, run_dir=None, seed=config.SEED, device=None):
    # Reviewer supplementary experiments (#1/#2): carrier normalization and the
    # low-band carrier override are wired through the FAST_GPU train cache
    # (_build_img_cache). The CPU train path does not honor them, so fail loud
    # rather than silently run an un-normalized baseline.
    _cn = getattr(config, "CARRIER_NORM", "off") != "off"
    _lb = abs(getattr(config, "LOWBAND_GHZ", 10.0) - 10.0) > 1e-9
    if (_cn or _lb) and not config.USE_FAST_GPU:
        raise NotImplementedError(
            "CARRIER_NORM / LOWBAND_GHZ require V921_FAST_GPU=1 (GPU train path)")
    if _cn:
        log(f"[carrier_norm] mode={config.CARRIER_NORM} "
            f"ref={config.CARRIER_NORM_REF_GHZ:.2f}GHz (train+eval, deterministic)")
    if _lb:
        log(f"[lowband] source 10GHz -> {config.LOWBAND_GHZ:.2f}GHz (DAS physics)")
    train_full = load_manifest("train", keep_7c=True)
    val_full = load_manifest("val", keep_7c=True)
    dev77_df, test77_final_df, split_info = load_or_build_test77_dev_final(run_dir=run_dir)

    if run_dir is not None:
        manifest_dir = run_dir / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        write_v9_2_1_manifests(
            manifest_dir,
            dev77_df=dev77_df,
            test77_final_df=test77_final_df,
            split_info=split_info,
        )

    if config.USE_FAST_GPU:
        source_loader = GpuSourceProvider(train_full, seed, device, log=log)
    else:
        source_loader = make_source_train_loader(train_full, seed=seed)
    val_10 = val_full[val_full["frequency"] == "10GHz"].reset_index(drop=True)
    val_24 = val_full[val_full["frequency"] == "24GHz"].reset_index(drop=True)

    log(
        f"train_source={len(train_full)} "
        f"train_src_rows=10:{(train_full['frequency'] == '10GHz').sum()} "
        f"24:{(train_full['frequency'] == '24GHz').sum()} "
        f"source_steps/epoch={len(source_loader)} "
        f"val_all={len(val_full)} heldout77_unused={len(dev77_df)} "
        f"test77_final={len(test77_final_df)}"
    )
    log(
        f"[test77_split] seed={split_info['seed']} "
        f"dev_per_class={split_info['dev_per_class']}"
    )
    if config.USE_FAST_GPU:
        val_all_loader = build_gpu_eval_loader(val_full, device)
        test77_loader = build_gpu_eval_loader(test77_final_df, device)
        dev77_loader = build_gpu_eval_loader(dev77_df, device)
    else:
        val_all_loader = make_eval_loader(val_full)
        test77_loader = make_eval_loader(test77_final_df)
        dev77_loader = make_eval_loader(dev77_df)
    return {
        "train_source": source_loader,
        "train_source_df": train_full,
        "val_all": val_all_loader,
        "val_10": make_eval_loader(val_10),
        "val_24": make_eval_loader(val_24),
        "test77_final": test77_loader,
        "dev77": dev77_loader,
        "val_all_df": val_full,
        "val_10_df": val_10,
        "val_24_df": val_24,
        "dev77_df": dev77_df,
        "test77_final_df": test77_final_df,
        "test77_split_info": split_info,
    }


def cosine_lr_schedule(step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def next_or_restart(loader, iterator):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def uniform_kl_loss(logits):
    target = torch.full_like(logits, 1.0 / logits.size(1))
    return F.kl_div(F.log_softmax(logits, dim=1), target, reduction="batchmean")


def soft_kl(logits_student, logits_teacher, temp=2.0, reduction="batchmean"):
    log_p = F.log_softmax(logits_student / temp, dim=1)
    q = F.softmax(logits_teacher.detach() / temp, dim=1)
    return F.kl_div(log_p, q, reduction=reduction) * (temp ** 2)


def decorrelation_loss(z_cls, z_freq):
    if z_cls.size(0) < 2:
        return z_cls.new_zeros(())
    a = z_cls - z_cls.mean(dim=0, keepdim=True)
    b = z_freq - z_freq.mean(dim=0, keepdim=True)
    a = F.normalize(a, dim=0)
    b = F.normalize(b, dim=0)
    cov = a.t() @ b / max(1, z_cls.size(0) - 1)
    return cov.pow(2).mean()


def carrier_residual_loss(model, z_cls, z_freq, residual):
    zero = z_cls.new_zeros(())
    if z_freq is None or residual is None or not hasattr(model, "predict_freq_residual"):
        return zero, zero, zero
    freq_weight = float(getattr(config, "V13_FREQ_WEIGHT", 0.0))
    decorr_weight = float(getattr(config, "V13_DECORR_WEIGHT", 0.0))
    if freq_weight <= 0.0 and decorr_weight <= 0.0:
        return zero, zero, zero
    pred_freq = model.predict_freq_residual(z_freq)
    loss_freq = F.smooth_l1_loss(pred_freq, residual.float())
    loss_decorr = decorrelation_loss(z_cls, z_freq)
    weighted = freq_weight * loss_freq + decorr_weight * loss_decorr
    return weighted, loss_freq, loss_decorr


def class_cone_loss(model, z, labels, domains):
    weight = float(getattr(config, "V16_CONE_WEIGHT", 0.0))
    if weight <= 0:
        return z.new_zeros(())
    eps = float(getattr(config, "V16_CONE_EPS", 0.75))
    temp = float(getattr(config, "V16_CONE_TEMP", 2.0))
    min_count = int(getattr(config, "V16_CONE_MIN_PER_BAND", 1))
    losses = []
    z_detached = z.detach()
    for cls_idx in range(config.NUM_CLASSES):
        cls_mask = labels == cls_idx
        m10 = cls_mask & (domains == config.FREQ_TO_IDX["10GHz"])
        m24 = cls_mask & (domains == config.FREQ_TO_IDX["24GHz"])
        if int(m10.sum()) < min_count or int(m24.sum()) < min_count:
            continue
        delta = z_detached[m24].mean(dim=0) - z_detached[m10].mean(dim=0)
        delta = F.normalize(delta, dim=0) * eps
        idx = cls_mask.nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            continue
        base = model.logits_from_neck(z[idx], margin=False)
        plus = model.logits_from_neck(z[idx] + delta.view(1, -1), margin=False)
        minus = model.logits_from_neck(z[idx] - delta.view(1, -1), margin=False)
        losses.append(0.5 * soft_kl(plus, base, temp) + 0.5 * soft_kl(minus, base, temp))
    if not losses:
        return z.new_zeros(())
    return torch.stack(losses).mean()


def v18_domain_lambda(global_step, total_steps):
    p = global_step / max(1, total_steps)
    return float(getattr(config, "V18_DOMAIN_LAMBDA_MAX", 0.5)) * (
        2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0
    )


def unpack_source_batch(batch):
    if len(batch) == 5:
        x_src, y_src, d_src, x_aux, r_src = batch
    elif len(batch) == 4:
        x_src, y_src, d_src, fourth = batch
        if hasattr(fourth, "ndim") and fourth.ndim <= 1:
            x_aux = None
            r_src = fourth
        else:
            x_aux = fourth
            r_src = None
    else:
        x_src, y_src, d_src = batch
        x_aux = None
        r_src = None
    return x_src, y_src, d_src, x_aux, r_src


def source_qualified_threshold():
    metric = str(getattr(config, "SOURCE_QUALIFIED_METRIC", "f1")).lower()
    if metric == "acc":
        return float(
            getattr(
                config,
                "SOURCE_QUALIFIED_MIN_ACC",
                getattr(config, "SOURCE_QUALIFIED_MIN_F1", 0.90),
            )
        )
    return float(getattr(config, "SOURCE_QUALIFIED_MIN_F1", 0.90))


def source_qualified_metric():
    metric = str(getattr(config, "SOURCE_QUALIFIED_METRIC", "f1")).lower()
    if metric not in ("acc", "f1"):
        raise ValueError(f"unsupported SOURCE_QUALIFIED_METRIC={metric!r}")
    return metric


def source_qualified_basis():
    basis = str(getattr(config, "SOURCE_QUALIFIED_BASIS", "ema")).lower()
    if basis not in ("live", "ema"):
        raise ValueError(f"unsupported SOURCE_QUALIFIED_BASIS={basis!r}")
    return basis


def source_qualified_value(row, metric, basis=None):
    basis = source_qualified_basis() if basis is None else basis
    key = f"val_{metric}_{basis}"
    return float(row.get(key, 0.0) or 0.0)


def candidate_target_f1(row, basis):
    return float(row.get(f"test77_f1_{basis}", 0.0) or 0.0)


def candidate_record(row, threshold, metric, basis):
    threshold_metric = f"val_{metric}_{basis}"
    return {
        "epoch": int(row["epoch"]),
        "threshold_metric": threshold_metric,
        "threshold": float(threshold),
        "checkpoint_basis": basis,
        "source_val_acc_live": float(row.get("val_acc_live", 0.0) or 0.0),
        "source_val_f1_live": float(row.get("val_f1_live", 0.0) or 0.0),
        "source_val_acc_ema": float(row.get("val_acc_ema", 0.0) or 0.0),
        "source_val_f1_ema": float(row.get("val_f1_ema", 0.0) or 0.0),
        "test77_acc_live": float(row.get("test77_acc_live", 0.0) or 0.0),
        "test77_f1_live": float(row.get("test77_f1_live", 0.0) or 0.0),
        "test77_acc_ema": float(row.get("test77_acc_ema", 0.0) or 0.0),
        "test77_f1_ema": float(row.get("test77_f1_ema", 0.0) or 0.0),
        "test77_per_class_live": row.get("test77_per_class_live", {}),
        "test77_per_class_ema": row.get("test77_per_class_ema", {}),
    }


def write_source_qualified_candidates(run_dir, threshold, metric, basis, records, best_record):
    threshold_metric = f"val_{metric}_{basis}"
    payload = {
        "policy": (
            f"source-qualified diagnostic pool: include {basis} checkpoints with "
            f"source validation {basis} {metric} >= threshold; rank recorded "
            f"candidates by 77GHz final-test {basis} macro-F1 only as an oracle "
            "diagnostic."
        ),
        "threshold_metric": threshold_metric,
        "threshold": float(threshold),
        "checkpoint_basis": basis,
        "target_labels_used_for_candidate_ranking": True,
        "best_by_test77_f1": best_record,
        "candidates": records,
    }
    (run_dir / "reports" / "source_qualified_candidates.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def train(args, run_dir, device, log):
    set_seed(args.seed)
    log(f"set_seed({args.seed})")
    loaders = build_dataloaders(log, run_dir=run_dir, seed=args.seed, device=device)
    source_loader = loaders["train_source"]

    model = TimmBackboneV921(
        args.backbone, config.NUM_CLASSES, adapter_mode=args.adapter_mode
    ).to(device)
    miro = MIROProjector(config.HEAD_HIDDEN, model.enc_dim).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable += list(miro.parameters())

    domain_clf = None
    if config.USE_DANN:
        domain_clf = DomainClassifier(
            config.HEAD_HIDDEN,
            hidden=config.DANN_HIDDEN,
            num_domains=config.NUM_FREQ_DOMAINS,
        ).to(device)
        trainable += list(domain_clf.parameters())

    # V13-as-GRL adversarial carrier-invariance head on z_cls (env-gated; OFF by
    # default). Regresses the continuous log-carrier through a gradient-reversal
    # layer so the encoder is pushed to make z_cls carrier-uninformative.
    carrier_adv = None
    grl_band_log_ghz = None
    grl_discrete = bool(getattr(config, "V13_GRL_DISCRETE", False))
    grl_bins = int(getattr(config, "V13_GRL_BINS", 3))
    grl_bin_edges = None          # interior edges (K-1,) for torch.bucketize
    if float(getattr(config, "V13_GRL_WEIGHT", 0.0)) > 0.0:
        carrier_adv = CarrierAdversary(
            config.HEAD_HIDDEN,
            hidden=int(getattr(config, "V13_GRL_HIDDEN", 128)),
            out_dim=(grl_bins if grl_discrete else 1),
        ).to(device)
        trainable += list(carrier_adv.parameters())
        # per-band source carrier in GHz indexed by d_src (0->10GHz, 1->24GHz)
        from v9_2_1lib import parse_freq_ghz as _pf
        grl_band_log_ghz = torch.log(
            torch.tensor([float(_pf(f)) for f in config.TRAIN_FREQS],
                         dtype=torch.float32, device=device)
        )
        if grl_discrete:
            # Discrete domain-adversary control: equal-width bins over the DAS
            # curriculum log-carrier range. The full edge set has grl_bins+1 points;
            # bucketize uses the grl_bins-1 INTERIOR edges (log f_eff below the first
            # -> bin 0, at/above the last -> bin K-1).
            from v9_2_1lib import das_log_carrier_range as _dlcr
            _lo, _hi = _dlcr()
            _full_edges = torch.linspace(_lo, _hi, grl_bins + 1, device=device)
            grl_bin_edges = _full_edges[1:-1].contiguous()
            _edges_ghz = [round(float(math.exp(e)), 2) for e in _full_edges.tolist()]
            log(f"V13_GRL_DISCRETE=1 bins={grl_bins} CE-on-carrier-bin; "
                f"log f_eff range=[{_lo:.4f},{_hi:.4f}] "
                f"bin edges (GHz)={_edges_ghz}")

    n_trainable = sum(p.numel() for p in trainable)
    extras = []
    if config.USE_DANN:
        extras.append("dann_head")
    if hasattr(model, "total_logits_from_neck"):
        extras.append("v15_falsification")
    if (
        hasattr(model, "encode_split")
        and (
            float(getattr(config, "V13_FREQ_WEIGHT", 0.0)) > 0
            or float(getattr(config, "V13_DECORR_WEIGHT", 0.0)) > 0
        )
    ):
        extras.append("v13_residual_split")
    if carrier_adv is not None:
        extras.append("v13_grl_carrier_adv")
    if (float(getattr(config, "V15R_FALSIFY_WEIGHT", 0.0)) > 0
            or float(getattr(config, "V15R_MARGIN_WEIGHT", 0.0)) > 0
            or float(getattr(config, "V15R_REALISM_WEIGHT", 0.0)) > 0
            or bool(getattr(config, "V15R_SINGLE_HEAD", False))):
        extras.append(
            "v15r["
            + ("realism," if float(getattr(config, "V15R_REALISM_WEIGHT", 0.0)) > 0 else "")
            + ("worstcase," if bool(getattr(config, "V15R_WORSTCASE", False)) else "")
            + ("margin," if float(getattr(config, "V15R_MARGIN_WEIGHT", 0.0)) > 0 else "")
            + ("singlehead" if bool(getattr(config, "V15R_SINGLE_HEAD", False)) else "")
            + "]"
        )
    if float(getattr(config, "V16_CONE_WEIGHT", 0.0)) > 0:
        extras.append("v16_class_cone")
    if hasattr(model, "grammar_counterfactual_loss"):
        extras.append("v17_grammar")
    if hasattr(model, "inverse_measurement_loss"):
        extras.append("v18_inverse_measurement")
    extras_label = ("+" + "+".join(extras)) if extras else ""
    log(
        f"trainable params={n_trainable} "
        f"(encoder_{args.adapter_mode}+neck+head + miro_proj{extras_label})"
    )

    opt = torch.optim.AdamW(trainable, lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    ce = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    ce_domain = nn.CrossEntropyLoss()

    logit_prior = compute_logit_prior(loaders["train_source_df"], config.NUM_CLASSES, device)
    log(f"logit_prior={logit_prior.cpu().tolist()}")
    log(
        f"use_dann={config.USE_DANN}"
        + (f" weight={config.DANN_WEIGHT} hidden={config.DANN_HIDDEN}" if config.USE_DANN else "")
    )
    log(f"use_das={config.USE_DAS}")
    if config.USE_DAS:
        log(f"das_mode={getattr(config, 'DAS_MODE', 'curriculum')} initial_stage={das_stage_for_epoch(1)}")
    # Provenance for the P0-B aug ablation: report the (env-gated) generic radar
    # augmentation flags so the ablation row is self-documenting in train.log.
    log(f"aug_flags: USE_HFT={config.USE_HFT} USE_SPEC_AUGMENT={config.USE_SPEC_AUGMENT}")
    log(f"adapter_mode={args.adapter_mode}")

    ema = ModelEMA(model, config.EMA_DECAY)
    steps_per_epoch = len(source_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = config.WARMUP_EPOCHS * steps_per_epoch

    hist = []
    monitor_rows = []
    source_qualified_records = []
    source_qualified_threshold_f1 = source_qualified_threshold()
    source_qualified_metric_name = source_qualified_metric()
    source_qualified_basis_name = source_qualified_basis()
    best_sourceval_metric_ema = -1.0
    best_sourcequalified77_f1_ema = -1.0
    best_sourcequalified_record = None
    best_sourceval_ema_ckpt_path = (
        run_dir / "checkpoints" / f"{safe_name(args.backbone)}_best_sourceval_{source_qualified_basis_name}.pt"
    )
    best_sourcequalified77_ema_ckpt_path = (
        run_dir / "checkpoints" / f"{safe_name(args.backbone)}_best_sourcequalified77_{source_qualified_basis_name}.pt"
    )
    last_ckpt_path = run_dir / "checkpoints" / f"{safe_name(args.backbone)}_last.pt"

    imgs_per_step = config.SOURCE_BATCH_SIZE
    total_images = total_steps * imgs_per_step
    log(
        f"schedule: epochs={args.epochs} steps/epoch={steps_per_epoch} "
        f"imgs/step={imgs_per_step} total_steps={total_steps} "
        f"total_imgs={total_images} warmup_steps={warmup_steps}"
    )
    log(
        f"source_qualified_policy: val_{source_qualified_metric_name}_{source_qualified_basis_name}>="
        f"{source_qualified_threshold_f1:.2f} "
        "candidates are recorded; best candidate by 77GHz F1 is saved as diagnostic"
    )

    total_bar = tqdm(
        total=total_steps,
        desc=f"{config.EXPERIMENT_NAME}",
        leave=True,
        dynamic_ncols=True,
        unit="step",
        mininterval=0.2,
        smoothing=0.1,
    )

    global_step = 0
    imgs_seen = 0
    start_epoch = 1
    if getattr(args, "resume", None):
        # weights_only=False: our own bundle contains optimizer state + RNG
        # (numpy/python) objects, which torch>=2.6 refuses under the default
        # weights_only=True. Trusted (we wrote it).
        R = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(
            {k: v.to(device) for k, v in R["model_trainable"].items()}, strict=False)
        ema.load_state_dict({k: v.to(device) for k, v in R["ema"].items()})
        miro.load_state_dict({k: v.to(device) for k, v in R["miro"].items()})
        if domain_clf is not None and R.get("domain_clf") is not None:
            domain_clf.load_state_dict({k: v.to(device) for k, v in R["domain_clf"].items()})
        opt.load_state_dict(R["opt"])
        global_step = int(R["global_step"])
        start_epoch = int(R["epoch"]) + 1
        try:
            import random as _random
            import numpy as _np
            torch.set_rng_state(R["rng"]["torch"].cpu() if hasattr(R["rng"]["torch"], "cpu") else R["rng"]["torch"])
            if R["rng"].get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([s.cpu() if hasattr(s, "cpu") else s for s in R["rng"]["cuda"]])
            _np.random.set_state(R["rng"]["numpy"])
            _random.setstate(R["rng"]["python"])
        except Exception as _e:
            log(f"[resume] rng restore skipped: {_e}")
        log(
            f"[resume] loaded {args.resume}: completed_ep={R['epoch']} "
            f"global_step={global_step} -> continue to ep{args.epochs}. "
            f"LR re-targeted to new horizon total_steps={total_steps} "
            f"(resumed lr starts ~{cosine_lr_schedule(global_step, total_steps, warmup_steps, config.LR):.2e})"
        )

    pool_state = pool_protocol.init_pool(run_dir, log_fn=log)
    log(
        f"[pool] init threshold={pool_state['threshold']:.2f} "
        f"period={pool_state['period']} selection={pool_state['selection_metric']}"
    )
    for ep in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        source_loader.dataset.set_epoch(ep)
        das_stage = source_loader.dataset.current_das_stage()

        model.train()
        model.set_encoder_train_mode()
        miro.train()
        log(
            f"[train] epoch {ep}/{args.epochs} steps={steps_per_epoch} "
            f"use_dann={int(config.USE_DANN)} das={das_stage['name']} "
            f"f=[{das_stage['f_low']:.1f},{das_stage['f_high']:.1f}] "
            f"p={das_stage['p']:.2f} gpu={gpu_mem()}"
        )

        run_ce = run_sc = run_mi = run_dn = 0.0
        run_v13 = 0.0
        run_grl = 0.0
        run_v15r = 0.0
        run_v15 = run_v16 = run_v17 = run_v18 = 0.0
        grl_lambda_last = 0.0
        n_batches = 0
        correct_labeled = total_labeled = 0
        dann_lambda_last = 0.0
        v18_lambda_last = 0.0

        source_iter = iter(source_loader)
        for step in range(steps_per_epoch):
            lr_now = cosine_lr_schedule(global_step, total_steps, warmup_steps, config.LR)
            for g in opt.param_groups:
                g["lr"] = lr_now

            batch, source_iter = next_or_restart(source_loader, source_iter)
            x_src, y_src, d_src, x_aux, r_src = unpack_source_batch(batch)

            x_src = x_src.to(device, non_blocking=True)
            y_src = y_src.to(device, non_blocking=True)
            d_src = d_src.to(device, non_blocking=True)
            if x_aux is not None:
                x_aux = x_aux.to(device, non_blocking=True)
            if r_src is not None:
                r_src = r_src.to(device, non_blocking=True).float()

            opt.zero_grad(set_to_none=True)

            if config.USE_DANN:
                p = global_step / max(1, total_steps)
                dann_lambda = float(2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)
            else:
                dann_lambda = 0.0
            dann_lambda_last = dann_lambda

            with torch.autocast(device_type=autocast_device_type(device), dtype=amp_dtype()):
                if hasattr(model, "encode_split"):
                    z_oracle_src, z_neck_src, z_freq_src = model.encode_split(x_src)
                else:
                    z_oracle_src, z_neck_src = model.encode(x_src)
                    z_freq_src = None

                if hasattr(model, "total_logits_from_neck"):
                    # V15R single-head: the sensor head is a verified no-op, so
                    # train CE on the kinematic head only. Default (=0) uses the
                    # kin+sensor total head exactly as before (inert).
                    _use_total = not bool(getattr(config, "V15R_SINGLE_HEAD", False))
                    if _use_total:
                        raw_logits_src = model.total_logits_from_neck(z_neck_src, margin=False)
                        logits_adj = (
                            model.total_logits_from_neck(z_neck_src, y_src, margin=True)
                            - config.LOGIT_ADJUST_TAU * logit_prior
                        )
                    else:
                        raw_logits_src = model.logits_from_neck(z_neck_src, margin=False)
                        logits_adj = (
                            model.logits_from_neck(z_neck_src, y_src, margin=True)
                            - config.LOGIT_ADJUST_TAU * logit_prior
                        )
                    loss_ce = ce(logits_adj, y_src)
                    loss_kin = ce(
                        model.logits_from_neck(z_neck_src, y_src, margin=True)
                        - config.LOGIT_ADJUST_TAU * logit_prior,
                        y_src,
                    )
                    loss_v15 = float(getattr(config, "V15_KIN_SOURCE_WEIGHT", 0.0)) * loss_kin
                    if x_aux is not None:
                        if config.SKIP_ORACLE:
                            z_aux = model.encode_adapted(x_aux)
                        else:
                            _, z_aux = model.encode(x_aux)
                        kin_aux = (
                            model.logits_from_neck(z_aux, y_src, margin=True)
                            - config.LOGIT_ADJUST_TAU * logit_prior
                        )
                        loss_aux_ce = ce(kin_aux, y_src)
                        sensor_aux = model.sensor_logits_from_neck(z_aux, margin=False)
                        loss_sensor_uniform = uniform_kl_loss(sensor_aux)
                        teacher = raw_logits_src.detach()
                        conf = F.softmax(teacher, dim=1).max(dim=1).values
                        kl_each = F.kl_div(
                            F.log_softmax(model.logits_from_neck(z_aux, margin=False) / 2.0, dim=1),
                            F.softmax(teacher / 2.0, dim=1),
                            reduction="none",
                        ).sum(dim=1) * 4.0
                        mask = conf >= float(getattr(config, "V15_CONSIST_CONF", 0.60))
                        loss_cons = kl_each[mask].mean() if mask.any() else z_neck_src.new_zeros(())
                        loss_v15 = loss_v15 + float(getattr(config, "V15_FALSIFY_WEIGHT", 0.0)) * loss_aux_ce
                        loss_v15 = loss_v15 + float(getattr(config, "V15_SENSOR_UNIFORM_WEIGHT", 0.0)) * loss_sensor_uniform
                        loss_v15 = loss_v15 + float(getattr(config, "V15_CONSIST_WEIGHT", 0.0)) * loss_cons
                else:
                    raw_logits_src = model.logits_from_neck(z_neck_src, margin=False)
                    logits_adj = (
                        model.logits_from_neck(z_neck_src, y_src, margin=True)
                        - config.LOGIT_ADJUST_TAU * logit_prior
                    )
                    loss_ce = ce(logits_adj, y_src)
                    loss_v15 = z_neck_src.new_zeros(())

                loss_sc = supcon_loss(z_neck_src, y_src)
                loss_mi = miro(z_neck_src, z_oracle_src)
                loss_v13, loss_v13_freq, loss_v13_decorr = carrier_residual_loss(
                    model, z_neck_src, z_freq_src, r_src
                )
                # V13-as-GRL: adversarial carrier-invariance on z_cls (z_neck_src).
                # The adversary regresses the CONTINUOUS log-carrier the sample
                # actually shows: log(f_eff) = log(f_src[d_src]) + r_src (r_src is
                # the per-sample DAS amount log(f_virt/f_src), 0 if no DAS).
                if carrier_adv is not None and r_src is not None:
                    p_grl = global_step / max(1, total_steps)
                    grl_lambda = float(2.0 / (1.0 + math.exp(-10.0 * p_grl)) - 1.0)
                    grl_lambda_last = grl_lambda
                    if str(getattr(config, "V13_GRL_TARGET", "shown")) == "base":
                        # target the ORIGINAL acquisition band (10 vs 24), even
                        # through DAS distortion -> directly scrub base carrier.
                        log_f_eff = grl_band_log_ghz[d_src]
                    else:
                        # target the EFFECTIVE (DAS'd) carrier the sample shows.
                        log_f_eff = grl_band_log_ghz[d_src] + r_src.float()
                    adv_out = carrier_adv(grad_reverse(z_neck_src, grl_lambda))
                    if grl_discrete:
                        # discrete domain-adversary control: bin the SAME log f_eff
                        # the regressor sees, train the K-way head with cross-entropy.
                        bin_idx = torch.bucketize(
                            log_f_eff, grl_bin_edges
                        ).clamp_(0, grl_bins - 1)
                        loss_grl = F.cross_entropy(adv_out, bin_idx)
                    else:
                        loss_grl = F.smooth_l1_loss(adv_out.squeeze(-1), log_f_eff)
                else:
                    loss_grl = z_neck_src.new_zeros(())

                # V15R redesigned falsification: worst-case carrier + hardest-negative
                # margin on the KINEMATIC head (stacks on the GRL residual split).
                # Default weights 0 -> inert. The candidate views are pre-built by the
                # GpuSourceProvider; here we pick (per sample) the carrier that
                # MAXIMIZES the kinematic loss, then minimize CE + a hardest-negative
                # margin on that worst view.
                loss_v15r = z_neck_src.new_zeros(())
                _v15r_fw = float(getattr(config, "V15R_FALSIFY_WEIGHT", 0.0))
                _v15r_mw = float(getattr(config, "V15R_MARGIN_WEIGHT", 0.0))
                if (_v15r_fw > 0.0 or _v15r_mw > 0.0) and x_aux is not None:
                    cands = getattr(source_loader, "_last_aux_cands", None)
                    if bool(getattr(config, "V15R_WORSTCASE", False)) and cands is not None:
                        Bc, Kc = cands.shape[0], cands.shape[1]
                        with torch.no_grad():
                            z_c = model.encode_adapted(cands.reshape(Bc * Kc, *cands.shape[2:]))
                            kin_c = model.logits_from_neck(z_c, margin=False)
                            ce_c = F.cross_entropy(
                                kin_c, y_src.repeat_interleave(Kc), reduction="none"
                            ).view(Bc, Kc)
                            worst = ce_c.argmax(dim=1)
                        x_stress = cands[torch.arange(Bc, device=device), worst]
                    else:
                        x_stress = x_aux
                    z_stress = model.encode_adapted(x_stress)
                    # (i) keep the kinematic head correct under the worst-case carrier
                    kin_stress_adj = (
                        model.logits_from_neck(z_stress, y_src, margin=True)
                        - config.LOGIT_ADJUST_TAU * logit_prior
                    )
                    loss_wc_ce = ce(kin_stress_adj, y_src)
                    # (ii) hardest-negative margin: true logit >= hardest-neg + margin
                    kin_stress_raw = model.logits_from_neck(z_stress, margin=False).float()
                    true_logit = kin_stress_raw.gather(1, y_src.view(-1, 1)).squeeze(1)
                    neg = kin_stress_raw.clone()
                    neg.scatter_(1, y_src.view(-1, 1), float("-inf"))
                    hardest = neg.max(dim=1).values
                    margin_v = float(getattr(config, "V15R_MARGIN", 4.0))
                    loss_margin = F.relu(margin_v - (true_logit - hardest)).mean()
                    loss_v15r = _v15r_fw * loss_wc_ce + _v15r_mw * loss_margin

                loss_v16 = class_cone_loss(model, z_neck_src, y_src, d_src)
                if hasattr(model, "grammar_counterfactual_loss"):
                    loss_v17 = model.grammar_counterfactual_loss(z_neck_src, x_src, y_src)
                else:
                    loss_v17 = z_neck_src.new_zeros(())

                if hasattr(model, "inverse_measurement_loss"):
                    loss_inv = model.inverse_measurement_loss(z_neck_src, z_oracle_src, d_src)
                    v18_lambda = v18_domain_lambda(global_step, total_steps)
                    v18_lambda_last = v18_lambda
                    domain_logits = model.domain_logits(grad_reverse(z_neck_src, v18_lambda))
                    loss_v18_domain = ce_domain(domain_logits, d_src)
                    loss_v18 = (
                        float(getattr(config, "V18_RECON_WEIGHT", 0.0)) * loss_inv
                        + float(getattr(config, "V18_DOMAIN_WEIGHT", 0.0)) * loss_v18_domain
                    )
                else:
                    loss_v18 = z_neck_src.new_zeros(())

                loss_source = (
                    loss_ce
                    + config.SUPCON_WEIGHT * loss_sc
                    + config.MIRO_WEIGHT * loss_mi
                    + loss_v13
                    + float(getattr(config, "V13_GRL_WEIGHT", 0.0)) * loss_grl
                    + loss_v15
                    + loss_v15r
                    + float(getattr(config, "V16_CONE_WEIGHT", 0.0)) * loss_v16
                    + float(getattr(config, "V17_GRAMMAR_LOSS_WEIGHT", 0.0)) * loss_v17
                    + loss_v18
                )

                if config.USE_DANN and domain_clf is not None:
                    z_neck_src_rev = grad_reverse(z_neck_src, dann_lambda)
                    domain_logits = domain_clf(z_neck_src_rev)
                    loss_dn = ce_domain(domain_logits, d_src)
                    loss_source = loss_source + config.DANN_WEIGHT * loss_dn
                else:
                    loss_dn = z_neck_src.new_zeros(())

            loss_source.backward()
            opt.step()

            if ep >= config.EMA_START_EPOCH:
                ema.update(model)

            with torch.no_grad():
                pred_lab = raw_logits_src.argmax(1)
                correct_labeled += int((pred_lab == y_src).sum().item())
                total_labeled += y_src.numel()

            run_ce += float(loss_ce.detach())
            run_sc += float(loss_sc.detach())
            run_mi += float(loss_mi.detach())
            run_dn += float(loss_dn.detach())
            run_v13 += float(loss_v13.detach())
            run_grl += float(loss_grl.detach())
            run_v15r += float(loss_v15r.detach())
            run_v15 += float(loss_v15.detach())
            run_v16 += float(loss_v16.detach())
            run_v17 += float(loss_v17.detach())
            run_v18 += float(loss_v18.detach())
            n_batches += 1

            global_step += 1
            imgs_seen += x_src.size(0)

            total_bar.set_postfix_str(
                f"ep{ep:02d}/{args.epochs} step {step+1:03d}/{steps_per_epoch} "
                f"imgs {imgs_seen}/{total_images} ce={run_ce/n_batches:.3f} "
                f"sc={run_sc/n_batches:.3f} mi={run_mi/n_batches:.3f} "
                f"m={run_v13/n_batches + run_v15/n_batches + run_v16/n_batches + run_v17/n_batches + run_v18/n_batches:.3f} "
                f"acc={correct_labeled/max(1,total_labeled):.3f} "
                f"lr={lr_now:.1e} {gpu_mem()}",
                refresh=False,
            )
            total_bar.update(1)

            if (step + 1) == 1 or (step + 1) % config.LOG_EVERY_STEPS == 0 or (step + 1) == steps_per_epoch:
                log(
                    f"[train] ep={ep:02d} step={step+1:03d}/{steps_per_epoch} "
                    f"ce={run_ce/n_batches:.4f} sc={run_sc/n_batches:.4f} "
                    f"mi={run_mi/n_batches:.4f} dn={run_dn/max(1,n_batches):.4f} "
                    f"v13={run_v13/n_batches:.4f} grl={run_grl/n_batches:.4f} "
                    f"v15={run_v15/n_batches:.4f} v15r={run_v15r/n_batches:.4f} "
                    f"v16={run_v16/n_batches:.4f} "
                    f"v17={run_v17/n_batches:.4f} v18={run_v18/n_batches:.4f} "
                    f"acc={correct_labeled/max(1,total_labeled):.4f} "
                    f"lr={lr_now:.2e} lambda={dann_lambda_last:.3f} "
                    f"grl_lambda={grl_lambda_last:.3f} "
                    f"v18_lambda={v18_lambda_last:.3f} gpu={gpu_mem()}"
                )

        # Per-epoch diagnostic eval cadence. EVAL_EVERY>1 skips the val/test
        # live+EMA passes on most epochs to save ~4 ViT-L forwards/epoch; ep1, the
        # last epoch, and every POOL_PERIOD are always evaluated so the final-EMA
        # pool_ep100_ema.pt checkpoint (the reported result) is unaffected.
        _eval_every = int(getattr(config, "EVAL_EVERY", 1))
        _pool_period = int(pool_state.get("period", 10))
        do_eval = (
            _eval_every <= 1
            or ep == 1
            or ep == args.epochs
            or (ep % _eval_every == 0)
            or (ep % _pool_period == 0)
        )
        _zero = {"acc": 0.0, "macro_f1": 0.0, "per_class": {}}
        dev77_live = {"acc": 0.0, "macro_f1": 0.0, "per_class": {}}
        dev77_ema = {"acc": 0.0, "macro_f1": 0.0, "per_class": {}}
        if do_eval:
            val_live = evaluate(model, loaders["val_all"], device, amp_dtype())
            test77_live = evaluate(model, loaders["test77_final"], device, amp_dtype())
        else:
            val_live = dict(_zero)
            test77_live = dict(_zero)
        if do_eval and ep >= config.EMA_START_EPOCH:
            with SwapEMA(model, ema):
                val_ema = evaluate(model, loaders["val_all"], device, amp_dtype())
                test77_ema = evaluate(model, loaders["test77_final"], device, amp_dtype())
                pool_protocol.maybe_update_pool(
                    pool_state,
                    epoch=ep,
                    total_epochs=args.epochs,
                    val_ema_metrics=val_ema,
                    model=model,
                    ema=ema,
                    backbone=args.backbone,
                    evaluate_fn=evaluate,
                    dev77_loader=loaders["dev77"],
                    test77_final_loader=loaders["test77_final"],
                    device=device,
                    amp_dtype=amp_dtype(),
                    extra_meta={"adapter_mode": args.adapter_mode},
                )
        else:
            val_ema = {"acc": 0.0, "macro_f1": 0.0, "per_class": {}}
            test77_ema = {"acc": 0.0, "macro_f1": 0.0, "per_class": {}}

        row = {
            "epoch": ep,
            "time_s": time.time() - t0,
            "dann_lambda": dann_lambda_last,
            "v18_lambda": v18_lambda_last,
            "das_stage": das_stage["name"],
            "das_p": float(das_stage["p"]),
            "das_f_low": float(das_stage["f_low"]),
            "das_f_high": float(das_stage["f_high"]),
            "lr_end": current_lr(opt),
            "train_acc": correct_labeled / max(1, total_labeled),
            "loss_ce": run_ce / max(1, n_batches),
            "loss_sc": run_sc / max(1, n_batches),
            "loss_dn": run_dn / max(1, n_batches),
            "loss_mi": run_mi / max(1, n_batches),
            "loss_v13": run_v13 / max(1, n_batches),
            "loss_grl": run_grl / max(1, n_batches),
            "loss_v15r": run_v15r / max(1, n_batches),
            "loss_v15": run_v15 / max(1, n_batches),
            "loss_v16": run_v16 / max(1, n_batches),
            "loss_v17": run_v17 / max(1, n_batches),
            "loss_v18": run_v18 / max(1, n_batches),
            "val_acc_live": val_live["acc"],
            "val_f1_live": val_live["macro_f1"],
            "val_acc_ema": val_ema["acc"],
            "val_f1_ema": val_ema["macro_f1"],
            "dev77_acc_live": dev77_live["acc"],
            "dev77_f1_live": dev77_live["macro_f1"],
            "dev77_acc_ema": dev77_ema["acc"],
            "dev77_f1_ema": dev77_ema["macro_f1"],
            "dev77_per_class_live": dev77_live["per_class"],
            "dev77_per_class_ema": dev77_ema["per_class"],
            "test77_acc_live": test77_live["acc"],
            "test77_f1_live": test77_live["macro_f1"],
            "test77_per_class_live": test77_live["per_class"],
            "test77_acc_ema": test77_ema["acc"],
            "test77_f1_ema": test77_ema["macro_f1"],
            "test77_per_class_ema": test77_ema["per_class"],
        }
        hist.append(row)
        (run_dir / "reports" / "history.json").write_text(
            json.dumps(hist, indent=2),
            encoding="utf-8",
        )

        # Checkpoint-selection diagnostic (env-gated, default OFF -> zero
        # behaviour change). Dump every epoch's TRAINABLE-only weights (raw +
        # EMA, ~7.5MB each) so a post-hoc diagnostic can replay any
        # checkpoint-selection rule at full epoch resolution.
        if os.environ.get("DUMP_EPOCH_CKPTS", "0") == "1":
            _ep_dir = run_dir / "epoch_ckpts"
            _ep_dir.mkdir(parents=True, exist_ok=True)
            _tnames = set(model.trainable_parameter_names())
            _raw = {
                k: v.detach().to("cpu", torch.float32).clone()
                for k, v in model.state_dict().items() if k in _tnames
            }
            _ema = {
                k: v.detach().to("cpu", torch.float32).clone()
                for k, v in ema.state_dict().items()
            }
            atomic_save(
                {
                    "epoch": ep,
                    "raw": _raw,
                    "ema": _ema,
                    "backbone": args.backbone,
                    "adapter_mode": args.adapter_mode,
                },
                _ep_dir / f"ep{ep:03d}.pt",
            )

        # Full resumable bundle (env-gated). Overwrites run_dir/resume.pt every
        # 10 epochs and at the end so the latest is always available to continue
        # training (e.g. --resume run_dir/resume.pt --epochs 150).
        if os.environ.get("SAVE_RESUME", "0") == "1" and (ep % 10 == 0 or ep == args.epochs):
            save_resume_bundle(
                run_dir / "resume.pt",
                ep=ep, global_step=global_step,
                steps_per_epoch=steps_per_epoch, warmup_steps=warmup_steps,
                model=model, ema=ema, miro=miro, domain_clf=domain_clf,
                opt=opt, args=args,
            )
            log(f"[resume_save] ep{ep} global_step={global_step} -> {run_dir/'resume.pt'}")

        log(
            f"[ep_summary] ep={ep:02d}/{args.epochs} "
            f"val_live={val_live['acc']:.4f}/{val_live['macro_f1']:.4f} "
            f"val_ema={val_ema['acc']:.4f}/{val_ema['macro_f1']:.4f} "
            f"test77_live={test77_live['acc']:.4f}/{test77_live['macro_f1']:.4f} "
            f"test77_diag_ema={test77_ema['acc']:.4f}/{test77_ema['macro_f1']:.4f} "
            f"v13={run_v13/max(1,n_batches):.4f} grl={run_grl/max(1,n_batches):.4f} "
            f"v15={run_v15/max(1,n_batches):.4f} v15r={run_v15r/max(1,n_batches):.4f} "
            f"v16={run_v16/max(1,n_batches):.4f} "
            f"v17={run_v17/max(1,n_batches):.4f} v18={run_v18/max(1,n_batches):.4f} "
            f"das={das_stage['name']}[{das_stage['f_low']:.0f},{das_stage['f_high']:.0f}] "
            f"p={das_stage['p']:.2f} ep_time={time.time()-t0:.1f}s"
        )

        source_val_metric_ema = source_qualified_value(
            row,
            source_qualified_metric_name,
            source_qualified_basis_name,
        )
        if ep >= config.EMA_START_EPOCH and source_val_metric_ema > best_sourceval_metric_ema:
            best_sourceval_metric_ema = source_val_metric_ema
            payload = {
                "backbone": args.backbone,
                "epoch": ep,
                "selection_metric": (
                    f"source_val_{source_qualified_metric_name}_{source_qualified_basis_name}"
                ),
                "selected_value": best_sourceval_metric_ema,
                "checkpoint_basis": source_qualified_basis_name,
                "val_f1_live": val_live["macro_f1"],
                "val_acc_live": val_live["acc"],
                "val_f1": val_ema["macro_f1"],
                "val_acc": val_ema["acc"],
                "test77_f1_live": test77_live["macro_f1"],
                "test77_acc_live": test77_live["acc"],
                "test77_f1": test77_ema["macro_f1"],
                "test77_acc": test77_ema["acc"],
                "adapter_mode": args.adapter_mode,
                "experiment_name": config.EXPERIMENT_NAME,
                "use_dann": bool(config.USE_DANN),
            }
            if source_qualified_basis_name == "live":
                payload["model"] = model.state_dict()
            else:
                payload["ema"] = ema.state_dict()
            atomic_save(payload, best_sourceval_ema_ckpt_path)
            log(
                f"[save] best_sourceval_{source_qualified_basis_name} "
                f"val_{source_qualified_metric_name}_{source_qualified_basis_name}="
                f"{best_sourceval_metric_ema:.4f} "
                f"source_val_live={val_live['acc']:.4f}/{val_live['macro_f1']:.4f} "
                f"source_val_ema={val_ema['acc']:.4f}/{val_ema['macro_f1']:.4f} ep={ep}"
            )

        if (
            ep >= config.EMA_START_EPOCH
            and source_qualified_value(
                row,
                source_qualified_metric_name,
                source_qualified_basis_name,
            ) >= source_qualified_threshold_f1
        ):
            q_record = candidate_record(
                row,
                source_qualified_threshold_f1,
                source_qualified_metric_name,
                source_qualified_basis_name,
            )
            source_qualified_records.append(q_record)
            q_target_f1 = candidate_target_f1(q_record, source_qualified_basis_name)
            if q_target_f1 > best_sourcequalified77_f1_ema:
                best_sourcequalified77_f1_ema = q_target_f1
                best_sourcequalified_record = q_record
                payload = {
                    "backbone": args.backbone,
                    "epoch": ep,
                    "selection_metric": (
                        f"source_val_{source_qualified_metric_name}_{source_qualified_basis_name}"
                        f">={source_qualified_threshold_f1:.2f}"
                        f"_then_77GHz_{source_qualified_basis_name}_macro_f1"
                    ),
                    "source_qualified_threshold": source_qualified_threshold_f1,
                    "checkpoint_basis": source_qualified_basis_name,
                    "val_f1_live": q_record["source_val_f1_live"],
                    "val_acc_live": q_record["source_val_acc_live"],
                    "val_f1": q_record["source_val_f1_ema"],
                    "val_acc": q_record["source_val_acc_ema"],
                    "test77_f1_live": q_record["test77_f1_live"],
                    "test77_acc_live": q_record["test77_acc_live"],
                    "test77_f1": q_record["test77_f1_ema"],
                    "test77_acc": q_record["test77_acc_ema"],
                    "target_labels_used_for_candidate_ranking": True,
                    "adapter_mode": args.adapter_mode,
                    "experiment_name": config.EXPERIMENT_NAME,
                    "use_dann": bool(config.USE_DANN),
                }
                if source_qualified_basis_name == "live":
                    payload["model"] = model.state_dict()
                else:
                    payload["ema"] = ema.state_dict()
                atomic_save(payload, best_sourcequalified77_ema_ckpt_path)
                log(
                    f"[save] best_sourcequalified77_{source_qualified_basis_name} ep={ep} "
                    f"source_val_live={q_record['source_val_acc_live']:.4f}/"
                    f"{q_record['source_val_f1_live']:.4f} "
                    f"source_val_ema={q_record['source_val_acc_ema']:.4f}/"
                    f"{q_record['source_val_f1_ema']:.4f} "
                    f"test77_{source_qualified_basis_name}="
                    f"{q_record['test77_acc_' + source_qualified_basis_name]:.4f}/"
                    f"{q_record['test77_f1_' + source_qualified_basis_name]:.4f}"
                )
            write_source_qualified_candidates(
                run_dir,
                source_qualified_threshold_f1,
                source_qualified_metric_name,
                source_qualified_basis_name,
                source_qualified_records,
                best_sourcequalified_record,
            )

        if ep % 10 == 0 or ep == args.epochs:
            monitor_metric_key = (
                f"val_{source_qualified_metric_name}_{source_qualified_basis_name}"
            )
            candidates = [
                r for r in hist
                if float(r.get(monitor_metric_key, 0.0) or 0.0) > 0.0
            ]
            if candidates:
                best_row = max(
                    candidates,
                    key=lambda r: float(r.get(monitor_metric_key, 0.0) or 0.0),
                )
                qualified = [
                    r for r in hist
                    if source_qualified_value(
                        r,
                        source_qualified_metric_name,
                        source_qualified_basis_name,
                    ) >= source_qualified_threshold_f1
                ]
                best_q = (
                    max(
                        qualified,
                        key=lambda r: float(
                            r.get(f"test77_f1_{source_qualified_basis_name}", 0.0) or 0.0
                        ),
                    )
                    if qualified else None
                )
                mon = {
                    "monitor_epoch": ep,
                    "best_epoch_so_far": int(best_row["epoch"]),
                    "source_val_f1_live": float(best_row["val_f1_live"]),
                    "source_val_acc_live": float(best_row["val_acc_live"]),
                    "source_val_f1_ema": float(best_row["val_f1_ema"]),
                    "source_val_acc_ema": float(best_row["val_acc_ema"]),
                    "test77_f1_live_diagnostic": float(best_row["test77_f1_live"]),
                    "test77_acc_live_diagnostic": float(best_row["test77_acc_live"]),
                    "test77_f1_ema_diagnostic": float(best_row["test77_f1_ema"]),
                    "test77_acc_ema_diagnostic": float(best_row["test77_acc_ema"]),
                    "source_qualified_threshold": source_qualified_threshold_f1,
                    "source_qualified_metric": source_qualified_metric_name,
                    "source_qualified_basis": source_qualified_basis_name,
                    "source_qualified_count": len(qualified),
                    "best_source_metric_ema": float(best_row[monitor_metric_key]),
                }
                if best_q is not None:
                    mon.update({
                        "best_sourcequalified77_epoch_so_far": int(best_q["epoch"]),
                        "best_sourcequalified77_source_val_f1_live": float(best_q["val_f1_live"]),
                        "best_sourcequalified77_source_val_acc_live": float(best_q["val_acc_live"]),
                        "best_sourcequalified77_source_val_f1_ema": float(best_q["val_f1_ema"]),
                        "best_sourcequalified77_source_val_acc_ema": float(best_q["val_acc_ema"]),
                        "best_sourcequalified77_test77_f1_live": float(best_q["test77_f1_live"]),
                        "best_sourcequalified77_test77_acc_live": float(best_q["test77_acc_live"]),
                        "best_sourcequalified77_test77_f1_ema": float(best_q["test77_f1_ema"]),
                        "best_sourcequalified77_test77_acc_ema": float(best_q["test77_acc_ema"]),
                    })
                monitor_rows.append(mon)
                (run_dir / "reports" / "strict77_monitor.json").write_text(
                    json.dumps(monitor_rows, indent=2),
                    encoding="utf-8",
                )
                mon_source_acc = mon["source_val_acc_" + source_qualified_basis_name]
                mon_source_f1 = mon["source_val_f1_" + source_qualified_basis_name]
                mon_test_acc = mon["test77_acc_" + source_qualified_basis_name + "_diagnostic"]
                mon_test_f1 = mon["test77_f1_" + source_qualified_basis_name + "_diagnostic"]
                msg = (
                    f"[strict77_monitor] ep={ep:02d} best_source_epoch={mon['best_epoch_so_far']} "
                    f"source_val_{source_qualified_basis_name}="
                    f"{mon_source_acc:.4f}/{mon_source_f1:.4f} "
                    f"test77_{source_qualified_basis_name}_diag="
                    f"{mon_test_acc:.4f}/{mon_test_f1:.4f} "
                    f"sourcequalified_count={mon['source_qualified_count']}"
                )
                if best_q is not None:
                    mon_q_source_acc = mon[
                        "best_sourcequalified77_source_val_acc_" + source_qualified_basis_name
                    ]
                    mon_q_source_f1 = mon[
                        "best_sourcequalified77_source_val_f1_" + source_qualified_basis_name
                    ]
                    mon_q_test_acc = mon[
                        "best_sourcequalified77_test77_acc_" + source_qualified_basis_name
                    ]
                    mon_q_test_f1 = mon[
                        "best_sourcequalified77_test77_f1_" + source_qualified_basis_name
                    ]
                    msg += (
                        f" best_sourcequalified77_epoch={mon['best_sourcequalified77_epoch_so_far']} "
                        f"source_val_{source_qualified_basis_name}="
                        f"{mon_q_source_acc:.4f}/{mon_q_source_f1:.4f} "
                        f"test77_{source_qualified_basis_name}="
                        f"{mon_q_test_acc:.4f}/{mon_q_test_f1:.4f}"
                    )
                log(msg)

        if not config.SKIP_LAST_CKPT:
            atomic_save(
                {
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "backbone": args.backbone,
                    "epoch": ep,
                    "adapter_mode": args.adapter_mode,
                    "experiment_name": config.EXPERIMENT_NAME,
                    "use_dann": bool(config.USE_DANN),
                },
                last_ckpt_path,
            )

    pool_protocol.finalize_pool(pool_state)
    (run_dir / "reports" / "history.json").write_text(
        json.dumps(hist, indent=2),
        encoding="utf-8",
    )
    total_bar.close()
    log(
        f"[train] done best_sourceval_{source_qualified_metric_name}_{source_qualified_basis_name}="
        f"{best_sourceval_metric_ema:.4f}"
    )
    return {
        "best_sourceval_ema": best_sourceval_ema_ckpt_path,
        "best_sourcequalified77_ema": best_sourcequalified77_ema_ckpt_path,
        "last": last_ckpt_path,
        "loaders": loaders,
        "model": model,
        "ema": ema,
    }


def load_ckpt_into_model(model, ema, ckpt_path, which, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if which == "ema":
        if "ema" not in ckpt:
            raise ValueError(f"{ckpt_path} has no 'ema' state")
        ema.load_state_dict(ckpt["ema"])
        ema.copy_to(model)
    elif which == "live":
        if "model" not in ckpt:
            raise ValueError(f"{ckpt_path} has no 'model' state")
        model.load_state_dict(ckpt["model"])
    elif which == "last":
        model.load_state_dict(ckpt["model"])
        if "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
    else:
        raise ValueError(f"unknown ckpt selector {which!r}")


def freq_scan(model, df, device, log, label):
    results = {}
    for f_virt in config.FREQ_SCAN_POINTS:
        loader = make_eval_loader(df, fixed_virtual_freq=f_virt)
        r = evaluate(model, loader, device, amp_dtype())
        results[str(f_virt)] = {
            "acc": r["acc"],
            "macro_f1": r["macro_f1"],
            "per_class": r["per_class"],
        }
        log(f"[freq_scan:{label}] f_virt={f_virt:.0f}GHz acc={r['acc']:.4f} f1={r['macro_f1']:.4f}")
    return results


def eval_only(args, run_dir, device, log):
    model = TimmBackboneV921(
        args.backbone, config.NUM_CLASSES, adapter_mode=args.adapter_mode
    ).to(device)
    ema = ModelEMA(model, config.EMA_DECAY)
    sq_basis = source_qualified_basis()

    if args.ckpt == "sourceval":
        path = run_dir / "checkpoints" / f"{safe_name(args.backbone)}_best_sourceval_{sq_basis}.pt"
        selector = sq_basis
    elif args.ckpt == "sourcequalified":
        path = run_dir / "checkpoints" / f"{safe_name(args.backbone)}_best_sourcequalified77_{sq_basis}.pt"
        selector = sq_basis
    elif args.ckpt == "ema":
        path = run_dir / "checkpoints" / f"{safe_name(args.backbone)}_best_ema.pt"
        selector = "ema"
    elif args.ckpt == "best":
        path = run_dir / "checkpoints" / f"{safe_name(args.backbone)}_best_live.pt"
        selector = "live"
    else:
        path = run_dir / "checkpoints" / f"{safe_name(args.backbone)}_last.pt"
        selector = "last"

    if not path.exists() and args.ckpt in ("sourceval", "sourcequalified"):
        raise FileNotFoundError(f"requested checkpoint not found: {path}")
    if not path.exists():
        fallback = run_dir / "checkpoints" / f"{safe_name(args.backbone)}_last.pt"
        log(f"[eval] {path.name} missing; falling back to {fallback.name}")
        path = fallback
        selector = "last"

    load_ckpt_into_model(model, ema, path, selector, device)
    log(f"[eval] loaded {path.name} ({args.ckpt})")

    val_full = load_manifest("val", keep_7c=True)
    val_10 = val_full[val_full["frequency"] == "10GHz"].reset_index(drop=True)
    val_24 = val_full[val_full["frequency"] == "24GHz"].reset_index(drop=True)
    dev77_df, test77_final_df, split_info = load_or_build_test77_dev_final(run_dir=run_dir)

    sq_metric = source_qualified_metric()
    if args.ckpt == "sourcequalified":
        eval_policy = (
            f"source-qualified diagnostic selected "
            f"(source-val {sq_basis} {sq_metric} >= {source_qualified_threshold():.2f}, "
            f"then best 77GHz {sq_basis} macro-F1)"
        )
        uses_target_for_selection = True
    else:
        eval_policy = f"strict source-val selected by {sq_basis} {sq_metric}"
        uses_target_for_selection = False
    log(
        f"[eval] {eval_policy}; heldout77_unused={len(dev77_df)} "
        f"final_test77={len(test77_final_df)} imgs, seed={split_info['seed']} "
        f"dev_per_class={split_info['dev_per_class']}"
    )

    test_r = evaluate(model, make_eval_loader(test77_final_df), device, amp_dtype())
    log(f"[eval] 77GHz final-test acc={test_r['acc']:.4f} f1={test_r['macro_f1']:.4f}")
    log(f"[eval] 77GHz final-test per-class={test_r['per_class']}")
    val_r = evaluate(model, make_eval_loader(val_full), device, amp_dtype())
    log(f"[eval] val_all acc={val_r['acc']:.4f} f1={val_r['macro_f1']:.4f}")

    scan_10 = freq_scan(model, val_10, device, log, "val10")
    scan_24 = freq_scan(model, val_24, device, log, "val24")

    summary = {
        "checkpoint": path.name,
        "ckpt_selector": args.ckpt,
        "task": "known_people_unknown_frequency_strict_sourceval_selection",
        "adapter_mode": args.adapter_mode,
        "experiment_name": config.EXPERIMENT_NAME,
        "selection_proxy": {
            "name": (
                eval_policy if args.ckpt == "sourcequalified"
                else f"source validation {sq_basis} {sq_metric} (10+24GHz)"
            ),
            "seed": split_info["seed"],
            "dev_per_class": split_info["dev_per_class"],
            "target_labels_used_for_candidate_ranking": uses_target_for_selection,
        },
        "dev77": {"not_evaluated": True, "count": len(dev77_df)},
        "test77_final": {
            "acc": test_r["acc"],
            "macro_f1": test_r["macro_f1"],
            "per_class": test_r["per_class"],
        },
        "val_all": {
            "acc": val_r["acc"],
            "macro_f1": val_r["macro_f1"],
            "per_class": val_r["per_class"],
        },
        "freq_scan_from_val_10GHz": scan_10,
        "freq_scan_from_val_24GHz": scan_24,
    }
    out = run_dir / "reports" / f"eval_{args.ckpt}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log(f"[eval] saved {out.name}")
    write_report(run_dir, summary, args.backbone, args.adapter_mode)


def write_report(run_dir, summary, backbone, adapter_mode):
    lines = [
        f"# {config.EXPERIMENT_NAME}",
        "",
        f"- Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"- Backbone: {backbone}",
        f"- Adapter mode: {adapter_mode}",
        f"- Checkpoint: {summary['checkpoint']} ({summary['ckpt_selector']})",
        f"- Selection: {summary['selection_proxy']['name']}",
        "",
        "## 77GHz final test",
        "",
    ]
    t = summary["test77_final"]
    lines.append(f"- acc: {t['acc']:.4f}")
    lines.append(f"- macro-F1: {t['macro_f1']:.4f}")
    lines.append("")
    lines.append("### per-class")
    lines.append("")
    for c, v in t["per_class"].items():
        lines.append(f"- {c}: {v:.4f}")
    lines.append("")
    v = summary["val_all"]
    lines.append("## val (10+24GHz source validation)")
    lines.append("")
    lines.append(f"- acc: {v['acc']:.4f}")
    lines.append(f"- macro-F1: {v['macro_f1']:.4f}")
    lines.append("")
    for key, title in (
        ("freq_scan_from_val_10GHz", "Frequency scan on 10GHz val"),
        ("freq_scan_from_val_24GHz", "Frequency scan on 24GHz val"),
    ):
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| f_virt (GHz) | acc | macro-F1 |")
        lines.append("|---|---|---|")
        for f_virt, r in summary[key].items():
            lines.append(f"| {f_virt} | {r['acc']:.4f} | {r['macro_f1']:.4f} |")
        lines.append("")
    with open(run_dir / "reports" / "REPORT.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")


def main():
    args = parse_args()
    config.TERMINAL_PROGRESS = args.progress
    run_dir = make_run_dir(args)
    log, log_fp = log_factory(run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log(f"run_dir={run_dir}")
    log(f"device={device}")
    log(f"backbone={args.backbone}")
    log(
        f"env: manifest_dir={config.MANIFEST_DIR} dataset_root={config.DATASET_ROOT} "
        f"use_dann={config.USE_DANN} "
        f"selection=source_val_{source_qualified_metric()}_{source_qualified_basis()} "
        f"adapter_mode={args.adapter_mode}"
    )

    try:
        if args.stage == "train":
            train_result = train(args, run_dir, device, log)
            eval_only(
                argparse.Namespace(
                    stage="eval",
                    backbone=args.backbone,
                    run_dir=run_dir,
                    ckpt="sourceval",
                    progress=args.progress,
                    epochs=args.epochs,
                    adapter_mode=args.adapter_mode,
                ),
                run_dir,
                device,
                log,
            )
            sourcequalified_path = train_result.get("best_sourcequalified77_ema")
            if sourcequalified_path is not None and Path(sourcequalified_path).exists():
                eval_only(
                    argparse.Namespace(
                        stage="eval",
                        backbone=args.backbone,
                        run_dir=run_dir,
                        ckpt="sourcequalified",
                        progress=args.progress,
                        epochs=args.epochs,
                        adapter_mode=args.adapter_mode,
                    ),
                    run_dir,
                    device,
                    log,
                )
        else:
            eval_only(args, run_dir, device, log)
    finally:
        log_fp.close()


if __name__ == "__main__":
    main()
