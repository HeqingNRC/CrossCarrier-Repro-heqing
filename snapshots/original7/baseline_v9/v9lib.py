from __future__ import annotations
import importlib.util
import math
import os
import random
import re
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import config
ROOT = Path(__file__).resolve().parent.parent
_BASE_V8LIB = ROOT / 'baseline_v8' / 'v8lib.py'
_SPEC = importlib.util.spec_from_file_location('_baseline_v8_reuse', _BASE_V8LIB)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f'Unable to load shared V8 library from {_BASE_V8LIB}')
_BASE = importlib.util.module_from_spec(_SPEC)
import sys as _sys
# Register so DataLoader spawn workers (Windows) can unpickle classes that
# live in this dynamically-loaded module (e.g. PerImageStandardize).
_sys.modules[_SPEC.name] = _BASE
_SPEC.loader.exec_module(_BASE)
set_seed = _BASE.set_seed
parse_freq_ghz = _BASE.parse_freq_ghz
load_manifest = _BASE.load_manifest
_write_base_manifests = _BASE.write_v8_manifests
das_deterministic = _BASE.das_deterministic
HighFreqDopplerTexturize = _BASE.HighFreqDopplerTexturize
tensor_transform = _BASE.tensor_transform
DomainClassifier = _BASE.DomainClassifier
MIROProjector = _BASE.MIROProjector
TimmBackboneV9 = _BASE.TimmBackboneV8
supcon_loss = _BASE.supcon_loss
compute_logit_prior = _BASE.compute_logit_prior
dann_lambda = _BASE.dann_lambda
ModelEMA = _BASE.ModelEMA
SwapEMA = _BASE.SwapEMA
grad_reverse = _BASE.grad_reverse
make_eval_loader = _BASE.make_eval_loader
_TS_RE = re.compile('_(\\d{6,})')

def extract_timestamp(source_file: str) -> str:
    m = _TS_RE.search(str(source_file))
    if not m:
        raise ValueError(f'Cannot extract timestamp from {source_file!r}')
    return m.group(1)

def add_pair_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['pair_ts'] = df['source_file'].map(extract_timestamp)
    df['pair_key'] = df['subject'].astype(str) + '|' + df['class'].astype(str) + '|' + df['pair_ts'].astype(str)
    return df

def build_exact_pairs(df_source: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    df_source = add_pair_keys(df_source)
    df10 = df_source[df_source['frequency'] == '10GHz'].copy()
    df24 = df_source[df_source['frequency'] == '24GHz'].copy()
    groups10 = defaultdict(list)
    groups24 = defaultdict(list)
    for row in df10.sort_values(['pair_key', 'path'], kind='mergesort').to_dict('records'):
        groups10[row['pair_key']].append(row)
    for row in df24.sort_values(['pair_key', 'path'], kind='mergesort').to_dict('records'):
        groups24[row['pair_key']].append(row)
    pair_rows = []
    matched_10_rows = 0
    matched_24_rows = 0
    common_keys = sorted(set(groups10) & set(groups24))
    for key in common_keys:
        rows10 = groups10[key]
        rows24 = groups24[key]
        n = min(len(rows10), len(rows24))
        matched_10_rows += n
        matched_24_rows += n
        for i in range(n):
            r10 = rows10[i]
            r24 = rows24[i]
            if r10['class_idx_7c'] != r24['class_idx_7c']:
                raise ValueError(f'class mismatch inside pair key {key}')
            pair_rows.append({'pair_key': key, 'subject': r10['subject'], 'class': r10['class'], 'class_idx_7c': int(r10['class_idx_7c']), 'pair_ts': r10['pair_ts'], 'path_10': r10['path'], 'source_file_10': r10['source_file'], 'path_24': r24['path'], 'source_file_24': r24['source_file']})
    pair_df = pd.DataFrame(pair_rows).sort_values(['class', 'subject', 'pair_ts', 'path_10', 'path_24'], kind='mergesort').reset_index(drop=True)
    stats = {'rows_10': int(len(df10)), 'rows_24': int(len(df24)), 'unique_keys_10': int(df10['pair_key'].nunique()), 'unique_keys_24': int(df24['pair_key'].nunique()), 'matched_pairs': int(len(pair_df)), 'matched_10_rows': int(matched_10_rows), 'matched_24_rows': int(matched_24_rows), 'unmatched_10_rows': int(len(df10) - matched_10_rows), 'unmatched_24_rows': int(len(df24) - matched_24_rows), 'coverage_10': float(matched_10_rows / max(1, len(df10))), 'coverage_24': float(matched_24_rows / max(1, len(df24)))}
    return (pair_df, stats)

def flatten_pair_rows(pair_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in pair_df.to_dict('records'):
        rows.append({'class_idx_7c': int(row['class_idx_7c']), 'frequency': '10GHz'})
        rows.append({'class_idx_7c': int(row['class_idx_7c']), 'frequency': '24GHz'})
    return pd.DataFrame(rows)

def write_v9_manifests(out_dir: Path, pair_df: pd.DataFrame | None=None) -> None:
    _write_base_manifests(out_dir)
    if pair_df is not None:
        pair_df.to_csv(out_dir / 'train_pairs.csv', index=False)

def das_stage_for_epoch(epoch: int) -> dict[str, float | str]:
    if not config.USE_DAS:
        return {'name': 'off', 'p': 0.0, 'f_low': 0.0, 'f_high': 0.0}
    if epoch <= config.DAS_STAGE1_END_EPOCH:
        return {'name': 'stage1', 'p': config.DAS_STAGE1_P, 'f_low': config.DAS_STAGE1_F_LOW_GHZ, 'f_high': config.DAS_STAGE1_F_HIGH_GHZ}
    if epoch <= config.DAS_STAGE2_END_EPOCH:
        return {'name': 'stage2', 'p': config.DAS_STAGE2_P, 'f_low': config.DAS_STAGE2_F_LOW_GHZ, 'f_high': config.DAS_STAGE2_F_HIGH_GHZ}
    return {'name': 'stage3', 'p': config.DAS_STAGE3_P, 'f_low': config.DAS_STAGE3_F_LOW_GHZ, 'f_high': config.DAS_STAGE3_F_HIGH_GHZ}

class ExactPairDataset(Dataset):

    def __init__(self, pair_df: pd.DataFrame, train: bool=True):
        self.df = pair_df.reset_index(drop=True)
        self.train = train
        self.root = config.DATASET_ROOT
        self.transform = tensor_transform(train=train)
        self.hft = HighFreqDopplerTexturize(config.HFT_P, config.HFT_FLOOR_AMP, config.HFT_HF_AMP) if train and config.USE_HFT else None
        self.epoch = 1

    def __len__(self) -> int:
        return len(self.df)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def current_das_stage(self) -> dict[str, float | str]:
        return das_stage_for_epoch(self.epoch)

    def _sample_virtual_freq(self) -> float | None:
        if not self.train:
            return None
        stage = self.current_das_stage()
        if random.random() > float(stage['p']):
            return None
        lo = float(stage['f_low'])
        hi = float(stage['f_high'])
        return math.exp(random.uniform(math.log(lo), math.log(hi)))

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img10 = Image.open(self.root / row['path_10']).convert('RGB')
        img24 = Image.open(self.root / row['path_24']).convert('RGB')
        f_virt = self._sample_virtual_freq()
        if f_virt is not None:
            img10 = das_deterministic(img10, 10.0, f_virt)
            img24 = das_deterministic(img24, 24.0, f_virt)
        if self.hft is not None:
            img10 = self.hft(img10)
            img24 = self.hft(img24)
        x10 = self.transform(img10)
        x24 = self.transform(img24)
        y = int(row['class_idx_7c'])
        d10 = config.FREQ_TO_IDX['10GHz']
        d24 = config.FREQ_TO_IDX['24GHz']
        return (x10, y, d10, x24, y, d24)

def class_balanced_pair_weights(pair_df: pd.DataFrame) -> torch.Tensor:
    counts = pair_df['class_idx_7c'].value_counts().to_dict()
    weights = pair_df['class_idx_7c'].map(lambda c: 1.0 / max(1, counts.get(int(c), 1))).astype(float).values
    return torch.as_tensor(weights, dtype=torch.double)

def make_pair_train_loader(pair_df: pd.DataFrame, seed: int):
    ds = ExactPairDataset(pair_df, train=True)
    weights = class_balanced_pair_weights(pair_df)
    g = torch.Generator()
    g.manual_seed(seed)
    sampler = WeightedRandomSampler(weights, num_samples=len(pair_df), replacement=True, generator=g)
    return DataLoader(ds, batch_size=config.PAIR_BATCH_SIZE, sampler=sampler, num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True)

def pair_cosine_loss(z10: torch.Tensor, z24: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(z10, z24, dim=1)).mean()

def symmetric_kl_loss(logits_a: torch.Tensor, logits_b: torch.Tensor, temp: float | None=None) -> torch.Tensor:
    temp = float(temp or config.PAIR_KL_TEMP)
    log_pa = F.log_softmax(logits_a / temp, dim=1)
    log_pb = F.log_softmax(logits_b / temp, dim=1)
    pa = log_pa.exp()
    pb = log_pb.exp()
    loss_ab = F.kl_div(log_pa, pb, reduction='batchmean')
    loss_ba = F.kl_div(log_pb, pa, reduction='batchmean')
    return 0.5 * (loss_ab + loss_ba) * temp ** 2

@torch.no_grad()
def evaluate(model, loader, device, amp_dtype):
    model.eval()
    (preds, labels) = ([], [])
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    for batch in loader:
        (x, y, _d) = batch
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device_type, dtype=amp_dtype):
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
