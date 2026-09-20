"""Shared infrastructure for the NEW public-baseline table (PUBLIC_BASELINE_DESIGN.md v2).

ONE protocol for every method:
  - train bands 10 + 24 GHz, test ZERO-SHOT on the held-out 77 GHz band (418 imgs, 7 classes);
  - 100 epochs, report the EMA checkpoint at the FINAL epoch (no early stop, no target peek);
  - 3 seeds (42 / 1234 / 31415), mean +/- std;
  - input = 224x224x3 jet spectrogram, PER-IMAGE standardized (NOT ImageNet mean/std) --
    identical to baseline_v20/v9_2_1lib.tensor_transform(train=False);
  - same optimizer setup as ours for the generic backbones (AdamW lr3e-4 wd0.05, cosine + 3ep
    warmup, batch 16, bf16, EMA 0.999 from ep5, class-balanced WeightedRandomSampler,
    label smoothing 0.05) so any gap is METHOD, not training setup.

This module deliberately does NOT import `baselines/config.py` or `baselines/models.py`:
both define a top-level module named `config`, which collides with `baseline_v20/config.py`
(the one `v9_2_1lib` needs). We re-implement the tiny timm classifier wrapper here so a single
process can hold BOTH the generic backbones and the proposed TimmBackboneV921 without a clash.

GPU acceleration follows G:\\zhanghe\\GPU_TRAINING_SPEEDUP_PLAYBOOK.md: images are cached in
VRAM once (resize224 + /255) and per-image standardization runs on the GPU. Generic baselines
use NO DAS / HFT / SpecAugment (those are the method's contribution), so there is no grid_sample
in their path -- the GPU standardize is numerically identical to the CPU PerImageStandardize,
so absolute numbers are trustworthy (unlike the DAS fast-path, which the playbook warns about).
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# --- path setup: make `config` resolve to baseline_v20/config.py, then import the lib ---
REPO = Path(__file__).resolve().parents[3]            # .../Letter journal
V20 = REPO / "baseline_v20"
if str(V20) not in sys.path:
    sys.path.insert(0, str(V20))

import config                      # noqa: E402  -> baseline_v20/config.py
import v9_2_1lib as lib            # noqa: E402  -> wraps baseline_v9 -> baseline_v8
import portable_eval as portable

# The v9 import chain force-sets HF_HUB_OFFLINE=1 (so the large DINOv3 weights load
# from the local cache during training). The generic timm baselines, however, must
# fetch/verify their ImageNet weights from the Hub (vgg16_bn is not cached). Re-enable
# online mode for THIS pipeline only -- it does not touch the original training scripts,
# and cached models (incl. DINOv3) still resolve from disk. Mutate the already-read
# huggingface_hub constant too, since it is latched at first import.
import os                          # noqa: E402
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
try:
    import huggingface_hub.constants as _hfc   # noqa: E402
    _hfc.HF_HUB_OFFLINE = os.environ["HF_HUB_OFFLINE"] == "1"
except Exception:
    pass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = torch.bfloat16
NC = config.NUM_CLASSES
IMG = config.IMG_SIZE
SEEDS = [42, 1234, 31415]

# Generic image backbones (#1-6), timm names from the design (Tier 1, full fine-tune).
GENERIC_BACKBONES = {
    "vgg16_bn":      "vgg16_bn.tv_in1k",
    "mobilenetv3_l": "mobilenetv3_large_100.ra_in1k",
    "effnetb0":      "efficientnet_b0.ra_in1k",
    "convnext_t":    "convnext_tiny.fb_in1k",
    "swin_t":        "swin_tiny_patch4_window7_224.ms_in1k",
    "convnextv2_t":  "convnextv2_tiny.fcmae_ft_in1k",
}

GENERIC_LABELS = {
    "vgg16_bn":      "VGG16-BN",
    "mobilenetv3_l": "MobileNetV3-Large",
    "effnetb0":      "EfficientNet-B0",
    "convnext_t":    "ConvNeXt-Tiny",
    "swin_t":        "Swin-Tiny",
    "convnextv2_t":  "ConvNeXtV2-Tiny",
}

# Pre-registered strong-but-fair recipes for the public generic baselines.
# These are model-family stabilization choices, not target-domain tuning:
# - no DAS / no carrier-ratio augmentation / no ACR;
# - no target or source-val checkpoint selection;
# - all rows still report the final ep100 EMA weights.
GENERIC_STRONG_RECIPES = {
    # VGG has a huge classifier and overfits the tiny source set easily.
    "vgg16_bn": dict(
        lr=3e-4, backbone_lr=5e-5, head_lr=5e-4, weight_decay=0.01,
        dropout=0.50, head_hidden=512, freeze_backbone_epochs=10,
        label_smoothing=0.00, ema_decay=0.995, grad_clip=1.0,
        mixup_alpha=0.20, aug_intensity=0.10, aug_noise=0.015,
        aug_mask_prob=0.50, aug_mask_frac=0.08,
    ),
    # Small CNNs need enough adaptation while keeping the head more plastic.
    "mobilenetv3_l": dict(
        lr=3e-4, backbone_lr=1e-4, head_lr=7e-4, weight_decay=0.02,
        dropout=0.30, head_hidden=256, freeze_backbone_epochs=0,
        label_smoothing=0.02, ema_decay=0.995, grad_clip=1.0,
        mixup_alpha=0.15, aug_intensity=0.12, aug_noise=0.020,
        aug_mask_prob=0.50, aug_mask_frac=0.08,
    ),
    "effnetb0": dict(
        lr=3e-4, backbone_lr=8e-5, head_lr=7e-4, weight_decay=0.02,
        dropout=0.30, head_hidden=256, freeze_backbone_epochs=0,
        label_smoothing=0.02, ema_decay=0.995, grad_clip=1.0,
        mixup_alpha=0.15, aug_intensity=0.12, aug_noise=0.020,
        aug_mask_prob=0.50, aug_mask_frac=0.08,
    ),
    # Modern CNN/Transformer backbones are source-overfit prone on 644 images.
    "convnext_t": dict(
        lr=3e-4, backbone_lr=5e-5, head_lr=6e-4, weight_decay=0.05,
        dropout=0.35, head_hidden=512, freeze_backbone_epochs=5,
        label_smoothing=0.03, ema_decay=0.995, grad_clip=1.0,
        mixup_alpha=0.20, aug_intensity=0.10, aug_noise=0.015,
        aug_mask_prob=0.50, aug_mask_frac=0.08,
    ),
    "swin_t": dict(
        lr=3e-4, backbone_lr=4e-5, head_lr=6e-4, weight_decay=0.05,
        dropout=0.35, head_hidden=512, freeze_backbone_epochs=5,
        label_smoothing=0.03, ema_decay=0.995, grad_clip=1.0,
        mixup_alpha=0.15, aug_intensity=0.08, aug_noise=0.010,
        aug_mask_prob=0.40, aug_mask_frac=0.06,
    ),
    "convnextv2_t": dict(
        lr=3e-4, backbone_lr=4e-5, head_lr=6e-4, weight_decay=0.05,
        dropout=0.35, head_hidden=512, freeze_backbone_epochs=5,
        label_smoothing=0.03, ema_decay=0.995, grad_clip=1.0,
        mixup_alpha=0.20, aug_intensity=0.08, aug_noise=0.010,
        aug_mask_prob=0.40, aug_mask_frac=0.06,
    ),
}


# ---------------------------------------------------------------------------
# Generic timm classifier (replicates baselines/models.BaselineClassifier,
# minus the baselines/config dependency). Full fine-tune.
# ---------------------------------------------------------------------------
class GenericBackbone(nn.Module):
    """timm backbone (num_classes=0) -> trainable transfer head.

    Input is expected ALREADY per-image standardized (see make caches below).
    The whole network is trainable (full fine-tune); this matches the design's
    'generic CNN/Transformer baselines are fully fine-tuned' rule.
    """

    def __init__(
        self,
        backbone_key: str,
        num_classes: int = NC,
        dropout: float = 0.2,
        head_hidden: int = 0,
        pretrained: bool = True,
    ):
        super().__init__()
        import timm
        timm_name = GENERIC_BACKBONES[backbone_key]
        kwargs = dict(pretrained=pretrained, num_classes=0)
        if timm_name.startswith("vit_") or timm_name.startswith("eva"):
            kwargs["img_size"] = IMG
        try:
            self.backbone = timm.create_model(timm_name, **kwargs)
        except TypeError:
            kwargs.pop("img_size", None)
            self.backbone = timm.create_model(timm_name, **kwargs)
        self.backbone_key = backbone_key
        self.timm_name = timm_name
        with torch.no_grad():
            feat_dim = int(self.backbone(torch.zeros(1, 3, IMG, IMG)).shape[1])
        self.feat_dim = feat_dim
        self.head_hidden = int(head_hidden or 0)
        if self.head_hidden > 0:
            self.head = nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.Dropout(dropout),
                nn.Linear(feat_dim, self.head_hidden),
                nn.GELU(),
                nn.LayerNorm(self.head_hidden),
                nn.Dropout(dropout),
                nn.Linear(self.head_hidden, num_classes),
            )
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.Dropout(dropout),
                nn.Linear(feat_dim, num_classes),
            )

    def forward(self, x):
        return self.head(self.backbone(x).float())


# ---------------------------------------------------------------------------
# EMA (same semantics as baselines/train.py ModelEMA: shadow over ALL floating
# state_dict entries, i.e. params AND float buffers like BN running stats).
# ---------------------------------------------------------------------------
class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = config.EMA_DECAY):
        self.decay = float(decay)
        self.shadow = {n: p.detach().clone() for n, p in model.state_dict().items()
                       if p.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for n, p in model.state_dict().items():
            if not p.dtype.is_floating_point:
                continue
            if n not in self.shadow:
                self.shadow[n] = p.detach().clone()
            else:
                self.shadow[n].mul_(d).add_(p.detach(), alpha=1.0 - d)

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        sd = model.state_dict()
        for n, v in self.shadow.items():
            if n in sd:
                sd[n].copy_(v)

    def state_dict(self):
        return {n: v.detach().cpu() for n, v in self.shadow.items()}


# ---------------------------------------------------------------------------
# Optimization helpers
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_lr(step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


# ---------------------------------------------------------------------------
# Data: VRAM image caches (playbook). Eval cache is per-image standardized once
# (no random aug -> exact, zero-risk). Train cache is raw [0,1]; standardize per
# step. NO DAS/HFT/SpecAugment for the generic baselines.
# ---------------------------------------------------------------------------
def load_manifests():
    """train / val (10+24 source) and full-418 test (77 GHz, 7 classes)."""
    train = lib.load_manifest("train", keep_7c=True).reset_index(drop=True)
    val = lib.load_manifest("val", keep_7c=True).reset_index(drop=True)
    test = lib.load_manifest("test", keep_7c=True).reset_index(drop=True)
    return train, val, test


def build_train_cache(df):
    """imgs(raw [0,1], on DEVICE), ys, ds, fs -- reuses lib._build_img_cache."""
    return lib._build_img_cache(df, DEVICE)


def build_eval_batches(df, batch=None):
    """List of (x_std, y, d) GPU batches; preprocessing == tensor_transform(train=False)."""
    # Cache standardized tensors in CPU RAM, then move one batch at a time.
    return lib.build_gpu_eval_loader(df, torch.device("cpu"), batch=batch or portable.batch_size())


def class_balanced_weights(df, device):
    counts = df["class_idx_7c"].value_counts().to_dict()
    w = df["class_idx_7c"].map(lambda c: 1.0 / max(1, counts.get(int(c), 1))).astype("float64").values
    return torch.as_tensor(w, dtype=torch.double, device=device)


# ---------------------------------------------------------------------------
# Metric -- BYTE-FOR-BYTE the same code as
# baseline_v20/aggregate_ablation_finalema.py::eval_accf1 (manual per-class
# TP/FP/FN macro-F1). The forward is supplied by the caller so the SAME metric
# serves every model family (generic / proposed / external).
# ---------------------------------------------------------------------------
@torch.no_grad()
def preds_from_batches(forward_fn, batches):
    """forward_fn(x_std)->logits(B,NC); batches=list[(x_std,y,d)]. Returns (preds, ys) np arrays."""
    preds, ys = [], []
    for x, y, _d in batches:
        with portable.autocast(DEVICE):
            lg = forward_fn(x.to(DEVICE))
        preds.append(lg.float().argmax(1).cpu())
        ys.append(y.detach().cpu())
    return torch.cat(preds).numpy(), torch.cat(ys).numpy()


def acc_macrof1_perclass(preds, y):
    preds = np.asarray(preds); y = np.asarray(y)
    acc = float((preds == y).mean())
    f1s, ps, rs = [], [], []
    for c in range(NC):
        tp = int(((preds == c) & (y == c)).sum())
        fp = int(((preds == c) & (y != c)).sum())
        fn = int(((preds != c) & (y == c)).sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * p * r / (p + r) if (p + r) else 0.0)
        ps.append(p); rs.append(r)
    return acc, float(np.mean(f1s)), f1s, ps, rs


def eval_forward(forward_fn, batches):
    preds, y = preds_from_batches(forward_fn, batches)
    acc, mf1, f1s, ps, rs = acc_macrof1_perclass(preds, y)
    return {"acc": acc, "macro_f1": mf1, "per_class_f1": f1s,
            "per_class_p": ps, "per_class_r": rs,
            "preds": preds.tolist(), "labels": y.tolist()}


def confusion(preds, y):
    m = np.zeros((NC, NC), dtype=int)
    for t, p in zip(np.asarray(y), np.asarray(preds)):
        m[int(t), int(p)] += 1
    return m


# ---------------------------------------------------------------------------
# Bootstrap 95% CI over the 418 target clips (seeded; reproducible).
# ---------------------------------------------------------------------------
def bootstrap_ci_macrof1(preds, y, n_boot=2000, seed=12345):
    preds = np.asarray(preds); y = np.asarray(y); n = len(y)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        _a, f1, *_ = acc_macrof1_perclass(preds[idx], y[idx])
        vals[b] = f1
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), float(np.mean(vals))


# ---------------------------------------------------------------------------
# Latency: ms/sample, batch=1 bf16 (pattern from benchmark_inference.py).
# ---------------------------------------------------------------------------
@torch.no_grad()
def latency_ms_per_sample(forward_fn, warmup=20, iters=200):
    if DEVICE.type != "cuda":
        return None
    x1 = torch.randn(1, 3, IMG, IMG, device=DEVICE)
    with portable.autocast(DEVICE):
        for _ in range(warmup):
            _ = forward_fn(x1)
        torch.cuda.synchronize()
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        ts = []
        for _ in range(iters):
            starter.record(); _ = forward_fn(x1); ender.record()
            torch.cuda.synchronize(); ts.append(starter.elapsed_time(ender))
    return float(np.mean(ts)), float(np.std(ts))
