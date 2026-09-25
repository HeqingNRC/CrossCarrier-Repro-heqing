import math
import os
import random
import re
from pathlib import Path
try:
    import certifi
    os.environ['SSL_CERT_FILE'] = certifi.where()
    os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
    os.environ['CURL_CA_BUNDLE'] = certifi.where()
except Exception:
    pass
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import config
os.environ.setdefault('HF_HOME', str(config.WEIGHTS_DIR))
os.environ.setdefault('HUGGINGFACE_HUB_CACHE', str(config.WEIGHTS_DIR / 'hub'))
os.environ.setdefault('TORCH_HOME', str(config.WEIGHTS_DIR / 'torch'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
config.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
_FREQ_RE = re.compile('^\\s*(\\d+(?:\\.\\d+)?)\\s*GHz\\s*$', re.IGNORECASE)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_freq_ghz(s):
    m = _FREQ_RE.match(str(s))
    if not m:
        raise ValueError(f'cannot parse frequency {s!r}')
    return float(m.group(1))

def load_manifest(split, keep_7c=True):
    df = pd.read_csv(config.MANIFEST_DIR / f'{split}.csv')
    if keep_7c:
        df = df[df['class'].isin(config.CLASSES)].copy()
        df['class_idx_7c'] = df['class'].map(config.CLASS_TO_IDX).astype(int)
    df['freq_idx'] = df['frequency'].map(lambda f: config.FREQ_TO_IDX.get(f, -1)).astype(int)
    return df.reset_index(drop=True)

def split_77ghz_subject_stratified(df_77, per_class, seed):
    df_77 = df_77.reset_index(drop=True).copy()
    pivot = df_77.groupby(['subject', 'class']).size().unstack(fill_value=0)
    cls_cols = [c for c in config.CLASSES if c in pivot.columns]
    mins = pivot[cls_cols].min(axis=1).sort_values(kind='mergesort', ascending=False)
    single_held = None
    for (subj, m) in mins.items():
        if int(m) >= per_class:
            single_held = subj
            break
    info = {}
    if single_held is not None:
        (test_parts, unl_parts) = ([], [])
        for cls in config.CLASSES:
            sub = df_77[df_77['class'] == cls]
            held_df = sub[sub['subject'] == single_held].sort_values('path', kind='mergesort')
            other_df = sub[sub['subject'] != single_held]
            test_parts.append(held_df.iloc[:per_class])
            unl_parts.append(other_df)
            info[cls] = {'mode': 'single_subject', 'held_subject': str(single_held), 'held_subject_total_frames': int(len(held_df)), 'test_frames': per_class, 'dropped_frames': int(len(held_df) - per_class), 'unlabeled_train_frames': int(len(other_df))}
        df_test = pd.concat(test_parts, ignore_index=True)
        df_unl = pd.concat(unl_parts, ignore_index=True)
        assert single_held not in set(df_unl['subject']), 'held-out subject leaked into training'
    else:
        (test_parts, unl_parts) = ([], [])
        for cls in config.CLASSES:
            sub = df_77[df_77['class'] == cls]
            counts = sub.groupby('subject').size().reset_index(name='n').sort_values(['n', 'subject'], ascending=[False, True], kind='mergesort')
            held_subj = counts.iloc[0]['subject']
            held_n = int(counts.iloc[0]['n'])
            if held_n < per_class:
                raise ValueError(f'class {cls}: best subject {held_subj} has only {held_n} frames (< per_class={per_class})')
            held_df = sub[sub['subject'] == held_subj].sort_values('path', kind='mergesort')
            test_parts.append(held_df.iloc[:per_class])
            unl_parts.append(sub[sub['subject'] != held_subj])
            info[cls] = {'mode': 'per_class_subject', 'held_subject': str(held_subj), 'held_subject_total_frames': held_n, 'test_frames': per_class, 'dropped_frames': held_n - per_class, 'unlabeled_train_frames': int((sub['subject'] != held_subj).sum())}
        df_test = pd.concat(test_parts, ignore_index=True)
        df_unl = pd.concat(unl_parts, ignore_index=True)
    assert set(df_unl['path']).isdisjoint(set(df_test['path'])), 'train/test image leak after 77GHz split'
    _ = seed
    return (df_unl, df_test, info)

def write_v8_manifests(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'val', 'test'):
        df = load_manifest(split, keep_7c=True)
        df.to_csv(out_dir / f'{split}_7c.csv', index=False)

def _estimate_bg_color(img):
    arr = np.asarray(img)
    if arr.ndim == 2:
        corners = np.concatenate([arr[:5, :5].reshape(-1), arr[:5, -5:].reshape(-1), arr[-5:, :5].reshape(-1), arr[-5:, -5:].reshape(-1)])
        return int(np.median(corners))
    corners = np.concatenate([arr[:5, :5].reshape(-1, arr.shape[2]), arr[:5, -5:].reshape(-1, arr.shape[2]), arr[-5:, :5].reshape(-1, arr.shape[2]), arr[-5:, -5:].reshape(-1, arr.shape[2])], axis=0)
    return tuple((int(v) for v in np.median(corners, axis=0)))

class DopplerAxisStretch:

    def __init__(self, f_low_ghz, f_high_ghz, p=1.0):
        self.log_lo = math.log(f_low_ghz)
        self.log_hi = math.log(f_high_ghz)
        self.p = p

    def __call__(self, img, f_src_ghz):
        if random.random() > self.p:
            return img
        f_virt = math.exp(random.uniform(self.log_lo, self.log_hi))
        scale = f_virt / float(f_src_ghz)
        if abs(scale - 1.0) < 0.02:
            return img
        (w, h) = img.size
        new_h = max(1, int(round(h * scale)))
        if new_h == h:
            return img
        stretched = img.resize((w, new_h), Image.BILINEAR)
        if new_h > h:
            top = (new_h - h) // 2
            return stretched.crop((0, top, w, top + h))
        bg = _estimate_bg_color(img)
        canvas = Image.new(img.mode, (w, h), color=bg)
        top = (h - new_h) // 2
        canvas.paste(stretched, (0, top))
        return canvas

def das_deterministic(img, f_src_ghz, f_virt_ghz):
    scale = float(f_virt_ghz) / float(f_src_ghz)
    if abs(scale - 1.0) < 0.02:
        return img
    (w, h) = img.size
    new_h = max(1, int(round(h * scale)))
    if new_h == h:
        return img
    stretched = img.resize((w, new_h), Image.BILINEAR)
    if new_h > h:
        top = (new_h - h) // 2
        return stretched.crop((0, top, w, top + h))
    bg = _estimate_bg_color(img)
    canvas = Image.new(img.mode, (w, h), color=bg)
    top = (h - new_h) // 2
    canvas.paste(stretched, (0, top))
    return canvas

class HighFreqDopplerTexturize:

    def __init__(self, p=0.65, floor_amp=0.05, hf_amp=0.08, signal_thresh=0.15):
        self.p = p
        self.floor_amp = floor_amp
        self.hf_amp = hf_amp
        self.signal_thresh = signal_thresh

    def __call__(self, img):
        if random.random() > self.p:
            return img
        arr = np.asarray(img).astype(np.float32) / 255.0
        (h, w, _) = arr.shape
        rng = np.random.default_rng()
        arr = arr + rng.uniform(0.0, self.floor_amp, size=(h, w, 1)).astype(np.float32)
        signal = arr.max(axis=2, keepdims=True) > self.signal_thresh
        hf = rng.standard_normal((h, w, 1)).astype(np.float32) * self.hf_amp
        arr = np.clip(arr + hf * signal.astype(np.float32), 0.0, 1.0)
        return Image.fromarray((arr * 255.0).astype(np.uint8))

class PerImageStandardize:

    def __init__(self, eps=1e-06):
        self.eps = eps

    def __call__(self, t):
        mean = t.mean(dim=(1, 2), keepdim=True)
        std = t.std(dim=(1, 2), keepdim=True) + self.eps
        return (t - mean) / std

class RadarSpecAugment:

    def __init__(self, time_p, doppler_p, time_frac, doppler_frac):
        self.time_p = time_p
        self.doppler_p = doppler_p
        self.time_frac = time_frac
        self.doppler_frac = doppler_frac

    def __call__(self, x):
        (_, h, w) = x.shape
        if random.random() < self.time_p:
            width = max(1, int(w * random.uniform(0.03, self.time_frac)))
            left = random.randint(0, max(0, w - width))
            x[:, :, left:left + width] = 0.0
        if random.random() < self.doppler_p:
            height = max(1, int(h * random.uniform(0.03, self.doppler_frac)))
            top = random.randint(0, max(0, h - height))
            x[:, top:top + height, :] = 0.0
        return x

def tensor_transform(train):
    ops = [transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)), transforms.ToTensor(), PerImageStandardize()]
    if train and config.USE_SPEC_AUGMENT:
        ops.append(RadarSpecAugment(config.TIME_MASK_P, config.DOPPLER_MASK_P, config.TIME_MASK_FRAC, config.DOPPLER_MASK_FRAC))
    return transforms.Compose(ops)

class SpectrogramV8Dataset(Dataset):

    def __init__(self, df, transform, train=False, labeled=True, fixed_virtual_freq=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.train = train
        self.labeled = labeled
        self.root = config.DATASET_ROOT
        self.das = DopplerAxisStretch(config.DAS_F_LOW_GHZ, config.DAS_F_HIGH_GHZ, config.DAS_P) if train and config.USE_DAS else None
        self.hft = HighFreqDopplerTexturize(config.HFT_P, config.HFT_FLOOR_AMP, config.HFT_HF_AMP) if train and config.USE_HFT else None
        self.fixed_virtual_freq = fixed_virtual_freq

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.root / row['path']).convert('RGB')
        f_src = parse_freq_ghz(row['frequency'])
        if self.train and self.das is not None:
            img = self.das(img, f_src)
        if self.fixed_virtual_freq is not None:
            img = das_deterministic(img, f_src, self.fixed_virtual_freq)
        if self.train and self.hft is not None:
            img = self.hft(img)
        x = self.transform(img)
        if self.labeled:
            y = int(row['class_idx_7c'])
        else:
            y = -1
        d = int(row['freq_idx']) if row['freq_idx'] >= 0 else config.FREQ_TO_IDX.get(row['frequency'], -1)
        return (x, y, d)

def class_balanced_weights(df, num_classes):
    counts = df['class_idx_7c'].value_counts().to_dict()
    weights = df['class_idx_7c'].map(lambda c: 1.0 / max(1, counts.get(int(c), 1))).astype(float).values
    return torch.as_tensor(weights, dtype=torch.double)

def make_train_loader(df, labeled, seed):
    ds = SpectrogramV8Dataset(df, tensor_transform(train=True), train=True, labeled=labeled)
    if labeled:
        weights = class_balanced_weights(df, config.NUM_CLASSES)
        g = torch.Generator()
        g.manual_seed(seed)
        sampler = WeightedRandomSampler(weights, num_samples=len(df), replacement=True, generator=g)
        return DataLoader(ds, batch_size=config.PER_FREQ_BATCH, sampler=sampler, num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True)
    g = torch.Generator()
    g.manual_seed(seed + 1)
    return DataLoader(ds, batch_size=config.PER_FREQ_BATCH, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True, generator=g)

def make_eval_loader(df, fixed_virtual_freq=None, batch_size=None):
    ds = SpectrogramV8Dataset(df, tensor_transform(train=False), train=False, labeled=True, fixed_virtual_freq=fixed_virtual_freq)
    bs = batch_size or config.PER_FREQ_BATCH * config.NUM_FREQ_DOMAINS
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=False)

class ThreeStreamLoader:

    def __init__(self, loader_10, loader_24, loader_77):
        self.loaders = [loader_10, loader_24, loader_77]
        self.steps = max((len(l) for l in self.loaders))

    def __len__(self):
        return self.steps

    def __iter__(self):
        iters = [iter(l) for l in self.loaders]
        for _ in range(self.steps):
            out = []
            for (i, it) in enumerate(iters):
                try:
                    out.append(next(it))
                except StopIteration:
                    iters[i] = iter(self.loaders[i])
                    out.append(next(iters[i]))
            yield out

class GradReverse(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return (-ctx.lambda_ * grad_output, None)

def grad_reverse(x, lambda_):
    return GradReverse.apply(x, lambda_)

class DomainClassifier(nn.Module):

    def __init__(self, dim, num_domains, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, num_domains))

    def forward(self, z):
        return self.net(z)

class MIROProjector(nn.Module):

    def __init__(self, dim_in, dim_oracle):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_oracle)

    def forward(self, z_neck, z_oracle):
        pred = self.proj(z_neck)
        pred = F.normalize(pred, dim=1)
        target = F.normalize(z_oracle.detach(), dim=1)
        return (1.0 - (pred * target).sum(dim=1)).mean()

class TimmBackboneV8(nn.Module):

    def __init__(self, backbone_name, num_classes):
        super().__init__()
        import timm
        kwargs = dict(pretrained=config.PRETRAINED, num_classes=0)
        if 'dinov3' in backbone_name and backbone_name.startswith('vit_'):
            kwargs['img_size'] = config.IMG_SIZE
        self.encoder = timm.create_model(backbone_name, **kwargs)
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, config.IMG_SIZE, config.IMG_SIZE)
            self.enc_dim = int(self.encoder(dummy).shape[1])
        self.neck = nn.Sequential(nn.LayerNorm(self.enc_dim), nn.Dropout(config.FEATURE_DROPOUT), nn.Linear(self.enc_dim, config.HEAD_HIDDEN), nn.GELU(), nn.LayerNorm(config.HEAD_HIDDEN))
        self.weight = nn.Parameter(torch.randn(num_classes, config.HEAD_HIDDEN))
        nn.init.xavier_uniform_(self.weight)

    def encode(self, x):
        with torch.no_grad():
            z_oracle = self.encoder(x).float()
        z_neck = self.neck(z_oracle)
        return (z_oracle, z_neck)

    def logits_from_neck(self, z_neck, labels=None, margin=True):
        z = F.normalize(z_neck, dim=1)
        w = F.normalize(self.weight, dim=1)
        logits = z @ w.t()
        if labels is not None and margin:
            one_hot = F.one_hot(labels, num_classes=w.shape[0]).float()
            logits = logits - one_hot * config.ARC_MARGIN
        return logits * config.ARC_SCALE

    def forward(self, x, labels=None, margin=True):
        (_, z_neck) = self.encode(x)
        return self.logits_from_neck(z_neck, labels, margin)

def supcon_loss(features, labels, temp=None):
    if features.shape[0] < 2:
        return features.new_zeros(())
    temp = temp or config.SUPCON_TEMP
    features = F.normalize(features, dim=1)
    sim = features @ features.t() / temp
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.t()).float()
    self_mask = torch.eye(mask.size(0), device=mask.device)
    mask = mask - self_mask
    logits_mask = 1.0 - self_mask
    exp_sim = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-08)
    pos_count = mask.sum(dim=1)
    valid = pos_count > 0
    if not valid.any():
        return features.new_zeros(())
    per_sample = -(mask * log_prob).sum(dim=1) / pos_count.clamp_min(1.0)
    return per_sample[valid].mean()

def compute_logit_prior(df_labeled, num_classes, device):
    counts = df_labeled['class_idx_7c'].value_counts().to_dict()
    vec = np.array([counts.get(i, 0) for i in range(num_classes)], dtype=np.float64)
    vec = vec / max(1.0, vec.sum())
    vec = np.clip(vec, 0.0001, None)
    return torch.as_tensor(np.log(vec), dtype=torch.float32, device=device)

def dann_lambda(epoch):
    if epoch < config.DANN_RAMP_START_EPOCH:
        return 0.0
    if epoch >= config.DANN_RAMP_END_EPOCH:
        return config.DANN_LAMBDA_MAX
    span = max(1, config.DANN_RAMP_END_EPOCH - config.DANN_RAMP_START_EPOCH)
    frac = (epoch - config.DANN_RAMP_START_EPOCH) / span
    return config.DANN_LAMBDA_MAX * frac

@torch.no_grad()
def evaluate(model, loader, device, amp_dtype):
    model.eval()
    (preds, labels) = ([], [])
    for batch in loader:
        (x, y, _d) = batch
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type='cuda', dtype=amp_dtype):
            logits = model(x, margin=False)
        preds.extend(logits.argmax(1).cpu().tolist())
        labels.extend(y.tolist())
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    acc = float((preds == labels).mean()) if len(labels) else 0.0
    per_class = {}
    for (i, c) in enumerate(config.CLASSES):
        m = labels == i
        per_class[c] = float((preds[m] == i).mean()) if m.any() else 0.0
    f1s = []
    for i in range(config.NUM_CLASSES):
        tp = int(((preds == i) & (labels == i)).sum())
        fp = int(((preds == i) & (labels != i)).sum())
        fn = int(((preds != i) & (labels == i)).sum())
        if tp + fp == 0 or tp + fn == 0:
            f1s.append(0.0)
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    macro_f1 = float(np.mean(f1s))
    return {'acc': acc, 'per_class': per_class, 'macro_f1': macro_f1, 'preds': preds, 'labels': labels}

class ModelEMA:

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {}
        for (name, p) in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    def update(self, model):
        d = self.decay
        for (name, p) in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(d).add_(p.detach(), alpha=1.0 - d)

    def copy_to(self, model):
        msd = dict(model.named_parameters())
        for (name, v) in self.shadow.items():
            if name in msd:
                msd[name].data.copy_(v.data)

    def state_dict(self):
        return {k: v.detach().clone() for (k, v) in self.shadow.items()}

    def load_state_dict(self, sd):
        for (k, v) in sd.items():
            if k in self.shadow:
                self.shadow[k].data.copy_(v.data)

class SwapEMA:

    def __init__(self, model, ema):
        self.model = model
        self.ema = ema
        self.backup = None

    def __enter__(self):
        self.backup = {name: p.detach().clone() for (name, p) in self.model.named_parameters() if name in self.ema.shadow}
        self.ema.copy_to(self.model)
        return self.model

    def __exit__(self, exc_type, exc, tb):
        msd = dict(self.model.named_parameters())
        for (name, v) in self.backup.items():
            msd[name].data.copy_(v.data)
        self.backup = None
