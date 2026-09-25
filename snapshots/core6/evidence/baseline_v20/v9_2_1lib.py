"""Shared code for V9.2.1.

V9.2.1 keeps the V9.2 clean dev/test split protocol:
  - full 10/24GHz source batches drive CE/SupCon/MIRO
  - exact 10GHz <-> 24GHz pairs drive pair consistency only
  - 77GHz is split into a dev split for checkpoint selection and a final test
    split for final reporting only

The only modeling change is to let the backbone adapt slightly:
  - default: small LoRA on the feature extractor
  - optional: unfreeze the final transformer block (+ final norm)
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import random
from pathlib import Path

import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import config


ROOT = Path(__file__).resolve().parent.parent
_BASE_V9LIB = ROOT / "baseline_v9" / "v9lib.py"
_SPEC = importlib.util.spec_from_file_location("_baseline_v9_reuse", _BASE_V9LIB)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Unable to load shared V9 library from {_BASE_V9LIB}")
_BASE = importlib.util.module_from_spec(_SPEC)
import sys as _sys
# Register so DataLoader spawn workers (Windows) can unpickle classes that
# live in this dynamically-loaded module.
_sys.modules[_SPEC.name] = _BASE
_SPEC.loader.exec_module(_BASE)


set_seed = _BASE.set_seed
parse_freq_ghz = _BASE.parse_freq_ghz
load_manifest = _BASE.load_manifest
das_deterministic = _BASE.das_deterministic
HighFreqDopplerTexturize = _BASE.HighFreqDopplerTexturize
tensor_transform = _BASE.tensor_transform
MIROProjector = _BASE.MIROProjector
supcon_loss = _BASE.supcon_loss
compute_logit_prior = _BASE.compute_logit_prior
ModelEMA = _BASE.ModelEMA
SwapEMA = _BASE.SwapEMA
_BASE_make_eval_loader = _BASE.make_eval_loader


def make_eval_loader(df, fixed_virtual_freq=None, batch_size=None):
    """CPU eval loader. Family A (#1): when carrier normalization is enabled and
    the caller hasn't requested a specific virtual carrier, deterministically map
    every image to the common reference carrier (das scale = ref / f_src) so the
    final reported 77GHz test (eval_only) is on the SAME normalized input the
    model trained/selected on."""
    if fixed_virtual_freq is None and getattr(config, "CARRIER_NORM", "off") != "off":
        fixed_virtual_freq = float(config.CARRIER_NORM_REF_GHZ)
    return _BASE_make_eval_loader(df, fixed_virtual_freq=fixed_virtual_freq,
                                  batch_size=batch_size)


build_exact_pairs = _BASE.build_exact_pairs
write_v9_manifests = _BASE.write_v9_manifests
_BASE_das_stage_for_epoch = _BASE.das_stage_for_epoch


def das_stage_for_epoch(epoch: int) -> dict[str, float | str]:
    """V9.2.1 wrapper around the V9 DAS schedule.

    Adds support for non-curriculum DAS modes used by the schedule
    ablation. `curriculum` (default) is byte-identical to V9. The
    fixed_* modes apply DAS-as-operator throughout training.
    """
    if not config.USE_DAS:
        return {"name": "off", "p": 0.0, "f_low": 0.0, "f_high": 0.0}
    mode = getattr(config, "DAS_MODE", "curriculum")
    if mode == "curriculum":
        return _BASE_das_stage_for_epoch(epoch)
    if mode == "fixed_full":
        return {
            "name": "fixed_full",
            "p": config.DAS_FIXED_P,
            "f_low": config.DAS_FIXED_FULL_F_LOW_GHZ,
            "f_high": config.DAS_FIXED_FULL_F_HIGH_GHZ,
        }
    if mode == "fixed_narrow":
        return {
            "name": "fixed_narrow",
            "p": config.DAS_FIXED_P,
            "f_low": config.DAS_FIXED_NARROW_F_LOW_GHZ,
            "f_high": config.DAS_FIXED_NARROW_F_HIGH_GHZ,
        }
    if mode == "jitter":
        # physics-free axis jitter; rho is sampled in _sample_virtual_freq
        return {
            "name": "jitter",
            "p": getattr(config, "DAS_JITTER_P", 1.0),
            "f_low": -1.0,
            "f_high": -1.0,
        }
    raise ValueError(f"unknown DAS_MODE: {mode!r}")


def das_log_carrier_range() -> tuple[float, float]:
    """Global (min, max) NATURAL-LOG carrier (GHz) the DAS schedule can show.

    Used only to bin the DISCRETE carrier adversary (config.V13_GRL_DISCRETE=1).
    The adversary target is log f_eff = log f_src + r = log f_virt, where f_virt is
    sampled uniformly-in-log inside the active stage's [f_low, f_high]. The discrete
    bins must therefore cover the WIDEST carrier range the schedule visits:
      curriculum   -> union of the 3 stage bands = [min stage f_low, max stage f_high]
      fixed_full   -> [DAS_FIXED_FULL_F_LOW_GHZ,   DAS_FIXED_FULL_F_HIGH_GHZ]
      fixed_narrow -> [DAS_FIXED_NARROW_F_LOW_GHZ, DAS_FIXED_NARROW_F_HIGH_GHZ]
    Source-band carriers of no-DAS samples (f_eff in {10, 24} GHz) fall inside this
    range, so every sample maps to a valid bin.
    """
    mode = getattr(config, "DAS_MODE", "curriculum")
    if mode == "fixed_full":
        lo, hi = config.DAS_FIXED_FULL_F_LOW_GHZ, config.DAS_FIXED_FULL_F_HIGH_GHZ
    elif mode == "fixed_narrow":
        lo, hi = config.DAS_FIXED_NARROW_F_LOW_GHZ, config.DAS_FIXED_NARROW_F_HIGH_GHZ
    else:  # curriculum (default) and any other mode -> union of the 3 stage bands
        lo = min(config.DAS_STAGE1_F_LOW_GHZ, config.DAS_STAGE2_F_LOW_GHZ,
                 config.DAS_STAGE3_F_LOW_GHZ)
        hi = max(config.DAS_STAGE1_F_HIGH_GHZ, config.DAS_STAGE2_F_HIGH_GHZ,
                 config.DAS_STAGE3_F_HIGH_GHZ)
    return math.log(float(lo)), math.log(float(hi))


# Make pair loaders honour DAS_MODE too: the V9 base ExactPairDataset
# resolves `das_stage_for_epoch` from its own module globals, so we
# point that name at our wrapper.
_BASE.das_stage_for_epoch = das_stage_for_epoch


make_pair_train_loader = _BASE.make_pair_train_loader
pair_cosine_loss = _BASE.pair_cosine_loss
symmetric_kl_loss = _BASE.symmetric_kl_loss
evaluate = _BASE.evaluate


def _create_feature_encoder(backbone_name: str):
    import timm

    kwargs = dict(pretrained=config.PRETRAINED, num_classes=0)
    if "dinov3" in backbone_name and backbone_name.startswith("vit_"):
        kwargs["img_size"] = config.IMG_SIZE
    return timm.create_model(backbone_name, **kwargs)


def _wrap_lora(model: nn.Module) -> nn.Module:
    from peft import LoraConfig, get_peft_model

    lora_cfg = LoraConfig(
        r=config.LORA_RANK,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        target_modules=list(config.LORA_TARGET_MODULES),
        task_type=None,
    )
    return get_peft_model(model, lora_cfg)


def _unfreeze_last_block(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)
    if not hasattr(model, "blocks") or len(model.blocks) == 0:
        raise RuntimeError("Backbone has no transformer blocks; cannot unfreeze last block.")
    for p in model.blocks[-1].parameters():
        p.requires_grad_(True)
    if config.LAST_BLOCK_UNFREEZE_NORM and hasattr(model, "norm"):
        for p in model.norm.parameters():
            p.requires_grad_(True)


class TimmBackboneV921(nn.Module):
    """V20 backbone: V15 kinematic/sensor heads plus V13 residual branch."""

    def __init__(self, backbone_name, num_classes, adapter_mode=None):
        super().__init__()
        self.backbone_name = backbone_name
        self.adapter_mode = str(adapter_mode or config.BACKBONE_TUNE_MODE)
        if self.adapter_mode not in ("lora", "last_block", "full_ft", "frozen"):
            raise ValueError(f"Unsupported adapter_mode={self.adapter_mode!r}")

        self.oracle_encoder = _create_feature_encoder(backbone_name)
        for p in self.oracle_encoder.parameters():
            p.requires_grad_(False)
        self.oracle_encoder.eval()

        self.encoder = _create_feature_encoder(backbone_name)
        if self.adapter_mode == "lora":
            self.encoder = _wrap_lora(self.encoder)
        elif self.adapter_mode == "last_block":
            _unfreeze_last_block(self.encoder)
        elif self.adapter_mode == "frozen":
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        # full_ft: leave encoder fully trainable (timm default). All
        # backbone params receive gradient updates at the same LR as the
        # head -- this is the LoRA C1 ablation's "full fine-tune" rung.

        with torch.no_grad():
            dummy = torch.zeros(1, 3, config.IMG_SIZE, config.IMG_SIZE)
            self.enc_dim = int(self.oracle_encoder(dummy).shape[1])

        self.neck = nn.Sequential(
            nn.LayerNorm(self.enc_dim),
            nn.Dropout(config.FEATURE_DROPOUT),
            nn.Linear(self.enc_dim, config.HEAD_HIDDEN),
            nn.GELU(),
            nn.LayerNorm(config.HEAD_HIDDEN),
        )
        self.freq_neck = nn.Sequential(
            nn.LayerNorm(self.enc_dim),
            nn.Dropout(config.FEATURE_DROPOUT),
            nn.Linear(self.enc_dim, config.HEAD_HIDDEN),
            nn.GELU(),
            nn.LayerNorm(config.HEAD_HIDDEN),
        )
        self.freq_head = nn.Linear(config.HEAD_HIDDEN, 1)
        self.weight = nn.Parameter(torch.randn(num_classes, config.HEAD_HIDDEN))
        self.sensor_weight = nn.Parameter(torch.randn(num_classes, config.HEAD_HIDDEN))
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.sensor_weight)

    def set_encoder_train_mode(self):
        self.oracle_encoder.eval()
        self.encoder.train()

    def trainable_parameter_names(self):
        return [name for name, p in self.named_parameters() if p.requires_grad]

    def encode(self, x):
        with torch.no_grad():
            z_oracle = self.oracle_encoder(x).float()
        z_backbone = self.encoder(x).float()
        z_neck = self.neck(z_backbone)
        return z_oracle, z_neck

    def encode_split(self, x):
        with torch.no_grad():
            z_oracle = self.oracle_encoder(x).float()
        z_backbone = self.encoder(x).float()
        z_cls = self.neck(z_backbone)
        z_freq = self.freq_neck(z_backbone)
        return z_oracle, z_cls, z_freq

    def encode_adapted(self, x):
        """Adapter-only neck features, skipping the frozen oracle forward.

        The oracle output is only consumed by MIRO (on x_src). For the V15
        stress view x_aux and for evaluation it is computed and discarded, so
        this path is a bit-identical speedup when SKIP_ORACLE is enabled.
        """
        return self.neck(self.encoder(x).float())

    def predict_freq_residual(self, z_freq):
        return self.freq_head(z_freq).squeeze(1)

    def _metric_logits(self, z_neck, weight, labels=None, margin=True):
        z = F.normalize(z_neck, dim=1)
        w = F.normalize(weight, dim=1)
        logits = z @ w.t()
        if labels is not None and margin:
            one_hot = F.one_hot(labels, num_classes=w.shape[0]).float()
            logits = logits - one_hot * config.ARC_MARGIN
        return logits * config.ARC_SCALE

    def logits_from_neck(self, z_neck, labels=None, margin=True):
        return self._metric_logits(z_neck, self.weight, labels, margin)

    def sensor_logits_from_neck(self, z_neck, labels=None, margin=True):
        return self._metric_logits(z_neck, self.sensor_weight, labels, margin)

    def total_logits_from_neck(self, z_neck, labels=None, margin=True):
        kin = self.logits_from_neck(z_neck, labels, margin)
        sensor = self.sensor_logits_from_neck(z_neck, labels, margin=False)
        return kin + sensor

    def forward(self, x, labels=None, margin=True):
        _, z_neck = self.encode(x)
        if not margin:
            return self.logits_from_neck(z_neck, labels=None, margin=False)
        return self.logits_from_neck(z_neck, labels, margin)


class CurriculumSourceDataset(Dataset):
    """All labeled 10/24GHz source samples with epoch-aware DAS curriculum."""

    def __init__(self, df, train: bool = True):
        self.df = df.reset_index(drop=True)
        self.train = train
        self.root = config.DATASET_ROOT
        self.transform = tensor_transform(train=train)
        self.hft = HighFreqDopplerTexturize(
            config.HFT_P, config.HFT_FLOOR_AMP, config.HFT_HF_AMP
        ) if train and config.USE_HFT else None
        # epoch kept in shared memory so PERSISTENT DataLoader workers see the
        # DAS curriculum advance (set_epoch in the main process is visible to
        # already-spawned workers, which is what makes persistent_workers safe).
        self._epoch = torch.zeros(1, dtype=torch.long).share_memory_()
        self._epoch[0] = 1

    def __len__(self):
        return len(self.df)

    def set_epoch(self, epoch: int) -> None:
        self._epoch[0] = int(epoch)

    def current_das_stage(self) -> dict[str, float | str]:
        return das_stage_for_epoch(int(self._epoch.item()))

    def _sample_virtual_freq(self, f_src: float | None = None) -> float | None:
        if not self.train:
            return None
        stage = self.current_das_stage()
        if random.random() > float(stage["p"]):
            return None
        if stage.get("name") == "jitter":
            # physics-free: rho drawn directly, decoupled from carrier ratio.
            rho = random.uniform(
                float(config.DAS_JITTER_RHO_LOW),
                float(config.DAS_JITTER_RHO_HIGH),
            )
            return float(f_src) * rho
        lo = float(stage["f_low"])
        hi = float(stage["f_high"])
        return math.exp(random.uniform(math.log(lo), math.log(hi)))

    def _sample_falsify_freq(self) -> float:
        freqs = list(getattr(config, "V15_OOD_FREQS", [77.0, 99.0, 120.0, 140.0]))
        return float(random.choice(freqs))

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        base_img = Image.open(self.root / row["path"]).convert("RGB")
        f_src = parse_freq_ghz(row["frequency"])
        img = base_img.copy()

        f_virt = self._sample_virtual_freq(f_src)
        freq_residual = 0.0
        if f_virt is not None:
            img = das_deterministic(img, f_src, f_virt)
            freq_residual = math.log(float(f_virt) / float(f_src))
        if self.hft is not None:
            img = self.hft(img)

        x = self.transform(img)
        img_ood = das_deterministic(base_img.copy(), f_src, self._sample_falsify_freq())
        if self.hft is not None:
            img_ood = self.hft(img_ood)
        x_ood = self.transform(img_ood)
        y = int(row["class_idx_7c"])
        d = int(row["freq_idx"]) if int(row["freq_idx"]) >= 0 else config.FREQ_TO_IDX.get(row["frequency"], -1)
        return x, y, d, x_ood, torch.tensor(freq_residual, dtype=torch.float32)


class _GradReverseFn(torch.autograd.Function):
    """Standard DANN gradient reversal layer."""

    @staticmethod
    def forward(ctx, x, lambda_adv):
        ctx.lambda_adv = lambda_adv
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_adv * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_adv: float) -> torch.Tensor:
    return _GradReverseFn.apply(x, lambda_adv)


class DomainClassifier(nn.Module):
    """Binary domain classifier on top of neck features.

    Used only when V921_USE_DANN=1. Receives gradient-reversed neck
    features and predicts the source band (10 GHz vs 24 GHz). Trained
    jointly with the encoder so the encoder is pushed to produce
    band-invariant features.
    """

    def __init__(self, feat_dim: int, hidden: int = 128, num_domains: int = 2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden, num_domains),
        )

    def forward(self, x):
        return self.head(x)


class CarrierAdversary(nn.Module):
    """V13-as-GRL adversarial carrier head on z_cls (the kinematic neck feature).

    Two modes, selected by `out_dim`:
      out_dim == 1 (default, CONTINUOUS -- our method): regresses the CONTINUOUS
        log-carrier log(f_eff) from z_cls via smooth-L1. Byte-identical to the
        original head.
      out_dim == K > 1 (DISCRETE -- config.V13_GRL_DISCRETE=1): outputs K
        carrier-BIN logits, trained with cross-entropy on a binned log(f_eff)
        label. This is the ordinary discrete domain-adversary control.

    Either way it is fed grad_reverse(z_cls, lambda) so that minimizing its loss
    MAXIMIZES the encoder's carrier confusion: through the reversed gradient the
    encoder is pushed to make z_cls carrier-uninformative, while the head itself
    still learns to read whatever carrier signal remains.

    Only built/optimized when config.V13_GRL_WEIGHT>0; at inference z_cls is
    used directly by the classifier and this head is discarded (not in EMA),
    exactly like the DANN DomainClassifier.
    """

    def __init__(self, feat_dim: int, hidden: int = 128, out_dim: int = 1):
        super().__init__()
        self.out_dim = out_dim
        self.head = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z_cls):
        return self.head(z_cls)


def class_balanced_source_weights(df) -> torch.Tensor:
    counts = df["class_idx_7c"].value_counts().to_dict()
    weights = df["class_idx_7c"].map(
        lambda c: 1.0 / max(1, counts.get(int(c), 1))
    ).astype(float).values
    return torch.as_tensor(weights, dtype=torch.double)


def make_source_train_loader(df, seed: int):
    ds = CurriculumSourceDataset(df, train=True)
    weights = class_balanced_source_weights(df)
    g = torch.Generator()
    g.manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights, num_samples=len(df), replacement=True, generator=g,
    )
    return DataLoader(
        ds,
        batch_size=config.SOURCE_BATCH_SIZE,
        sampler=sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(config.NUM_WORKERS > 0),
        prefetch_factor=(4 if config.NUM_WORKERS > 0 else None),
    )


def pair_weight_scale(epoch: int) -> float:
    ep = int(epoch)
    if ep <= config.PAIR_FULL_WEIGHT_END_EPOCH:
        return 1.0
    if ep >= config.PAIR_DECAY_END_EPOCH:
        return float(config.PAIR_FINAL_SCALE)
    span = max(1, config.PAIR_DECAY_END_EPOCH - config.PAIR_FULL_WEIGHT_END_EPOCH)
    progress = (ep - config.PAIR_FULL_WEIGHT_END_EPOCH) / span
    return float(1.0 + (config.PAIR_FINAL_SCALE - 1.0) * progress)


def _stable_split_key(seed: int, cls: str, subject: str, path: str) -> str:
    token = f"{seed}|{cls}|{subject}|{path}"
    return hashlib.sha1(token.encode("utf-8")).hexdigest()


def _subject_dev_quotas(counts: dict[str, int], dev_total: int) -> dict[str, int]:
    items = [(str(subject), int(n)) for subject, n in counts.items() if int(n) > 0]
    if not items:
        raise ValueError("Cannot allocate dev quotas for an empty class.")

    total_rows = sum(n for _subject, n in items)
    if dev_total <= 0 or dev_total >= total_rows:
        raise ValueError(
            f"Invalid dev_total={dev_total} for class total_rows={total_rows}"
        )

    if dev_total <= len(items):
        ranked = sorted(items, key=lambda x: (-x[1], x[0]))
        return {
            subject: (1 if i < dev_total else 0)
            for i, (subject, _n) in enumerate(ranked)
        }

    quotas = {subject: 1 for subject, _n in items}
    remaining = dev_total - len(items)
    residual_total = sum(max(0, n - 1) for _subject, n in items)
    if remaining <= 0:
        return quotas

    floor_used = 0
    remainders = []
    for subject, n in items:
        residual = max(0, n - 1)
        if residual_total > 0:
            ideal_extra = remaining * residual / residual_total
        else:
            ideal_extra = 0.0
        floor_extra = min(residual, int(math.floor(ideal_extra)))
        quotas[subject] += floor_extra
        floor_used += floor_extra
        remainders.append((subject, residual, ideal_extra - floor_extra))

    left = remaining - floor_used
    if left > 0:
        for subject, _residual, _frac in sorted(
            remainders,
            key=lambda x: (-x[2], -(x[1]), x[0]),
        ):
            if left <= 0:
                break
            if quotas[subject] < counts[subject]:
                quotas[subject] += 1
                left -= 1

    if left > 0:
        for subject, n in sorted(items, key=lambda x: (-x[1], x[0])):
            if left <= 0:
                break
            while quotas[subject] < n and left > 0:
                quotas[subject] += 1
                left -= 1

    return quotas


def split_test77_dev_final(
    df_77: pd.DataFrame,
    dev_per_class: int | None = None,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    dev_per_class = int(dev_per_class or config.TEST77_DEV_PER_CLASS)
    seed = int(seed or config.TEST77_DEV_SPLIT_SEED)
    df_77 = df_77.reset_index(drop=True).copy()

    dev_parts = []
    test_parts = []
    split_info = {
        "seed": seed,
        "dev_per_class": dev_per_class,
        "classes": {},
    }

    for cls in config.CLASSES:
        cls_df = df_77[df_77["class"] == cls].copy()
        if len(cls_df) <= dev_per_class:
            raise ValueError(
                f"class {cls}: only {len(cls_df)} rows available for "
                f"dev_per_class={dev_per_class}"
            )

        counts = {
            str(subject): int(n)
            for subject, n in cls_df.groupby("subject").size().to_dict().items()
        }
        quotas = _subject_dev_quotas(counts, dev_per_class)

        dev_cls_parts = []
        test_cls_parts = []
        for subject in sorted(counts):
            subj_df = cls_df[cls_df["subject"] == subject].copy()
            subj_df["_split_key"] = subj_df["path"].map(
                lambda p: _stable_split_key(seed, cls, subject, str(p))
            )
            subj_df = subj_df.sort_values(
                ["_split_key", "path"], kind="mergesort"
            ).reset_index(drop=True)
            n_dev = int(quotas.get(subject, 0))
            dev_cls_parts.append(subj_df.iloc[:n_dev].drop(columns=["_split_key"]))
            test_cls_parts.append(subj_df.iloc[n_dev:].drop(columns=["_split_key"]))

        dev_cls = pd.concat(dev_cls_parts, ignore_index=True)
        test_cls = pd.concat(test_cls_parts, ignore_index=True)
        if len(dev_cls) != dev_per_class:
            raise RuntimeError(
                f"class {cls}: expected {dev_per_class} dev rows, got {len(dev_cls)}"
            )
        if not set(dev_cls["path"]).isdisjoint(set(test_cls["path"])):
            raise RuntimeError(f"class {cls}: dev/test image overlap detected")

        split_info["classes"][cls] = {
            "total_rows": int(len(cls_df)),
            "dev_rows": int(len(dev_cls)),
            "test_rows": int(len(test_cls)),
            "subject_counts": counts,
            "subject_dev_quotas": {str(k): int(v) for k, v in quotas.items()},
        }

        dev_parts.append(dev_cls)
        test_parts.append(test_cls)

    dev_df = pd.concat(dev_parts, ignore_index=True).sort_values(
        ["class", "subject", "path"], kind="mergesort"
    ).reset_index(drop=True)
    test_df = pd.concat(test_parts, ignore_index=True).sort_values(
        ["class", "subject", "path"], kind="mergesort"
    ).reset_index(drop=True)

    if not set(dev_df["path"]).isdisjoint(set(test_df["path"])):
        raise RuntimeError("Global 77GHz dev/final-test overlap detected")

    return dev_df, test_df, split_info


def write_v9_2_1_manifests(
    out_dir: Path,
    pair_df=None,
    dev77_df: pd.DataFrame | None = None,
    test77_final_df: pd.DataFrame | None = None,
    split_info: dict | None = None,
) -> None:
    write_v9_manifests(out_dir, pair_df=pair_df)
    if dev77_df is not None:
        dev77_df.to_csv(out_dir / "test77_dev_7c.csv", index=False)
    if test77_final_df is not None:
        test77_final_df.to_csv(out_dir / "test77_final_7c.csv", index=False)
    if split_info is not None:
        with open(out_dir / "test77_split_info.json", "w", encoding="utf-8") as f:
            json.dump(split_info, f, indent=2)


def load_or_build_test77_dev_final(run_dir: Path | None = None):
    manifest_dir = None if run_dir is None else (run_dir / "manifests")
    if manifest_dir is not None:
        dev_path = manifest_dir / "test77_dev_7c.csv"
        test_path = manifest_dir / "test77_final_7c.csv"
        info_path = manifest_dir / "test77_split_info.json"
        if dev_path.exists() and test_path.exists():
            dev_df = pd.read_csv(dev_path)
            test_df = pd.read_csv(test_path)
            if info_path.exists():
                split_info = json.loads(info_path.read_text(encoding="utf-8"))
            else:
                split_info = {
                    "seed": config.TEST77_DEV_SPLIT_SEED,
                    "dev_per_class": config.TEST77_DEV_PER_CLASS,
                }
            return dev_df, test_df, split_info

    full_test77 = load_manifest("test", keep_7c=True)
    dev_df, test_df, split_info = split_test77_dev_final(
        full_test77,
        dev_per_class=config.TEST77_DEV_PER_CLASS,
        seed=config.TEST77_DEV_SPLIT_SEED,
    )
    if manifest_dir is not None:
        manifest_dir.mkdir(parents=True, exist_ok=True)
        dev_df.to_csv(manifest_dir / "test77_dev_7c.csv", index=False)
        test_df.to_csv(manifest_dir / "test77_final_7c.csv", index=False)
        with open(manifest_dir / "test77_split_info.json", "w", encoding="utf-8") as f:
            json.dump(split_info, f, indent=2)
    return dev_df, test_df, split_info


# =====================================================================
# GPU acceleration path (opt-in via config.USE_FAST_GPU).
# See G:\zhanghe\GPU_TRAINING_SPEEDUP_PLAYBOOK.md. CPU path above is unchanged.
# =====================================================================

def gpu_das(imgs, scales):
    """Batched Doppler-axis stretch via grid_sample (vertical rescale + center
    crop/pad). For compression (scale<1) the padded band is filled with the
    per-image background colour (median of the 4 corners) to match the CPU
    das_deterministic bg-pad; plain zeros-pad would inject black and shift the
    per-image standardization. Exact no-op for scale>=1 (crop, all in-bounds)."""
    n, C = imgs.shape[0], imgs.shape[1]
    cs = 5
    corners = torch.cat([imgs[..., :cs, :cs], imgs[..., :cs, -cs:],
                         imgs[..., -cs:, :cs], imgs[..., -cs:, -cs:]], dim=-1)
    bg = corners.reshape(n, C, -1).median(dim=2).values.view(n, C, 1, 1)
    theta = torch.zeros(n, 2, 3, device=imgs.device, dtype=imgs.dtype)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0 / scales
    grid = F.affine_grid(theta, imgs.shape, align_corners=False)
    out = F.grid_sample(imgs - bg, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return out + bg


def gpu_hft(im, f_virt=None):
    """High-freq Doppler texturizer. V15R realism (env-gated): when
    config.V15R_REALISM_WEIGHT>0 and per-sample virtual carrier f_virt (B,) is
    given, the HF-texture amplitude is scaled by (f_virt/REF)**gamma so higher
    virtual carriers carry stronger/finer micro-Doppler texture. gamma=0 (default)
    or f_virt=None -> exact current fixed-amplitude behaviour."""
    if not config.USE_HFT:
        return im
    n, _, H, W = im.shape
    hf_amp = config.HFT_HF_AMP
    gamma = float(getattr(config, "V15R_REALISM_WEIGHT", 0.0))
    if gamma > 0.0 and f_virt is not None:
        ref = float(getattr(config, "V15R_HFT_REF_GHZ", 24.0))
        mult = (f_virt.to(im.device).float().clamp(min=1e-3) / ref) ** gamma
        hf_amp = config.HFT_HF_AMP * mult.view(n, 1, 1, 1)
    ap = (torch.rand(n, 1, 1, 1, device=im.device) < config.HFT_P).float()
    out = im + torch.rand(n, 1, H, W, device=im.device) * config.HFT_FLOOR_AMP
    sig = (out.amax(1, keepdim=True) > 0.15).float()
    out = (out + torch.randn(n, 1, H, W, device=im.device) * hf_amp * sig).clamp(0, 1)
    return ap * out + (1 - ap) * im


def gpu_standardize(t, eps=1e-6):
    return (t - t.mean((2, 3), keepdim=True)) / (t.std((2, 3), keepdim=True) + eps)


def gpu_specaugment(t):
    if not config.USE_SPEC_AUGMENT:
        return t
    n, _, H, W = t.shape
    for b in range(n):
        if random.random() < config.TIME_MASK_P:
            w = max(1, int(W * random.uniform(0.03, config.TIME_MASK_FRAC)))
            l = random.randint(0, W - w); t[b, :, :, l:l + w] = 0.0
        if random.random() < config.DOPPLER_MASK_P:
            h = max(1, int(H * random.uniform(0.03, config.DOPPLER_MASK_FRAC)))
            o = random.randint(0, H - h); t[b, :, o:o + h, :] = 0.0
    return t


def _build_img_cache(df, device):
    import numpy as np
    n = len(df)
    imgs = torch.empty(n, 3, config.IMG_SIZE, config.IMG_SIZE)
    ys = torch.empty(n, dtype=torch.long)
    ds = torch.empty(n, dtype=torch.long)
    fs = torch.empty(n)
    for i in range(n):
        row = df.iloc[i]
        im = Image.open(config.DATASET_ROOT / row["path"]).convert("RGB").resize(
            (config.IMG_SIZE, config.IMG_SIZE), Image.BILINEAR)
        imgs[i] = torch.from_numpy(np.asarray(im, dtype="float32").transpose(2, 0, 1) / 255.0)
        ys[i] = int(row["class_idx_7c"])
        di = int(row["freq_idx"]) if int(row["freq_idx"]) >= 0 else config.FREQ_TO_IDX.get(row["frequency"], -1)
        ds[i] = di
        fs[i] = parse_freq_ghz(row["frequency"])
    imgs = imgs.to(device); ys = ys.to(device); ds = ds.to(device); fs = fs.to(device)
    # Family B (#2): remap the nominal 10 GHz low-band source carrier to its true
    # impulse-UWB effective centre (config.LOWBAND_GHZ, e.g. 7.3) so the DAS ratio
    # f_d ∝ f_c uses the corrected carrier. 24/77 GHz are left untouched.
    if abs(config.LOWBAND_GHZ - 10.0) > 1e-9:
        fs = torch.where((fs - 10.0).abs() < 1e-6,
                         torch.full_like(fs, float(config.LOWBAND_GHZ)), fs)
    # Family A (#1): deterministic carrier normalization -- resample every image's
    # Doppler axis to a common reference carrier (scale = ref / f_src), the same
    # operator the DAS curriculum uses but applied deterministically by the KNOWN
    # carrier. Covers both the GPU train cache and the GPU eval loaders (both call
    # _build_img_cache). Exact no-op for scale>=1 (crop), bg-pad for scale<1.
    if getattr(config, "CARRIER_NORM", "off") != "off":
        imgs = gpu_das(imgs, float(config.CARRIER_NORM_REF_GHZ) / fs)
    return imgs, ys, ds, fs


class GpuSourceProvider:
    """Drop-in replacement for the source train loader: all augmentation on GPU
    from a VRAM-resident cache. Yields the same 5-tuple (x_src, y, d, x_aux, r)
    as CurriculumSourceDataset, already on `device`. Honors USE_DAS / DAS_MODE
    (curriculum / fixed_* / jitter) and class-balanced sampling.
    """

    def __init__(self, df, seed, device, log=None):
        self.device = device
        self.imgs, self.y, self.d, self.f_src = _build_img_cache(df, device)
        self.n = self.imgs.shape[0]
        self.batch = config.SOURCE_BATCH_SIZE
        counts = df["class_idx_7c"].value_counts().to_dict()
        w = df["class_idx_7c"].map(lambda c: 1.0 / max(1, counts.get(int(c), 1))).astype("float64").values
        self.weights = torch.as_tensor(w, dtype=torch.double, device=device)
        self.gen = torch.Generator(device=device); self.gen.manual_seed(int(seed))
        self._epoch = 1
        self.dataset = self  # so train loop's source_loader.dataset.set_epoch(...) works
        self.ood = list(getattr(config, "V15_OOD_FREQS", [77.0, 99.0, 120.0, 140.0]))
        # V15R worst-case carrier candidates (built only when enabled; the train
        # step picks, per sample, the candidate that maximizes the kinematic loss).
        self.wc_enabled = bool(getattr(config, "V15R_WORSTCASE", False))
        self.wc_freqs = list(getattr(config, "V15R_WC_FREQS", self.ood)) or self.ood
        self._last_aux_cands = None      # (B, K, C, H, W) or None
        self._last_wc_freqs = None       # (K,) tensor or None
        if log:
            mem = self.imgs.element_size() * self.imgs.nelement() / 1e9
            log(f"[fast_gpu] source cache {self.n} imgs ({mem:.2f}GB) on {device}")

    def __len__(self):
        return self.n // self.batch

    def set_epoch(self, e):
        self._epoch = int(e)

    def current_das_stage(self):
        return das_stage_for_epoch(self._epoch)

    def _main_scales(self, fs):
        B = fs.shape[0]
        scales = torch.ones(B, device=self.device)
        stage = self.current_das_stage()
        if not config.USE_DAS or float(stage["p"]) <= 0:
            return scales
        apply = torch.rand(B, device=self.device, generator=self.gen) < float(stage["p"])
        if stage.get("name") == "jitter":
            cand = (torch.rand(B, device=self.device, generator=self.gen)
                    * (config.DAS_JITTER_RHO_HIGH - config.DAS_JITTER_RHO_LOW) + config.DAS_JITTER_RHO_LOW)
        else:
            lo, hi = math.log(float(stage["f_low"])), math.log(float(stage["f_high"]))
            fv = torch.exp(torch.rand(B, device=self.device, generator=self.gen) * (hi - lo) + lo)
            cand = fv / fs
        return torch.where(apply, cand, scales)

    def __iter__(self):
        return self

    def __next__(self):
        idx = torch.multinomial(self.weights, self.batch, replacement=True, generator=self.gen)
        base = self.imgs[idx]; fs = self.f_src[idx]
        y = self.y[idx]; d = self.d[idx]
        scales = self._main_scales(fs)
        # main view: f_virt = the carrier the (DAS'd) sample actually shows = fs*scale
        x = gpu_specaugment(gpu_standardize(gpu_hft(gpu_das(base, scales), f_virt=fs * scales)))
        r = torch.log(scales)
        ofreq = torch.tensor([random.choice(self.ood) for _ in range(self.batch)], device=self.device)
        x_aux = gpu_specaugment(gpu_standardize(gpu_hft(gpu_das(base, ofreq / fs), f_virt=ofreq)))
        # V15R: build the worst-case carrier candidate views (carrier-scaled HFT),
        # left for the train step to score+select per sample. Off -> no overhead.
        if self.wc_enabled:
            cands = []
            for fk in self.wc_freqs:
                fkv = torch.full((self.batch,), float(fk), device=self.device)
                vk = gpu_specaugment(gpu_standardize(gpu_hft(gpu_das(base, fkv / fs), f_virt=fkv)))
                cands.append(vk)
            self._last_aux_cands = torch.stack(cands, dim=1)              # (B, K, C, H, W)
            self._last_wc_freqs = torch.tensor(self.wc_freqs, device=self.device)
        else:
            self._last_aux_cands = None
            self._last_wc_freqs = None
        return x, y, d, x_aux, r


def build_gpu_eval_loader(df, device, batch=None):
    """Cached big-batch eval 'loader': list of (x_std, y, d) GPU-tensor batches.
    Preprocessing == tensor_transform(train=False) (resize224 + per-image
    standardize, no DAS/HFT/SpecAugment) -> identical input to CPU eval. Drop-in
    iterable for evaluate()."""
    batch = batch or config.EVAL_BATCH
    imgs, ys, ds, _ = _build_img_cache(df, device)
    imgs = gpu_standardize(imgs)
    return [(imgs[i:i + batch], ys[i:i + batch], ds[i:i + batch])
            for i in range(0, imgs.shape[0], batch)]
