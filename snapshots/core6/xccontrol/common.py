"""File/protocol utilities. Does not import the training project on the login node."""
from __future__ import annotations
import csv, hashlib, importlib.util, json, os, sys
from collections import Counter
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = KIT / 'evidence'
CLASSES6 = ['Away', 'Bend', 'Kneel', 'Pick', 'Sit', 'Towards']
CLASSES7 = ['Away', 'Bend', 'Kneel', 'Pick', 'SStep', 'Sit', 'Towards']
FREQUENCIES = ['10GHz', '24GHz', '77GHz']


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def read_csv(path: Path) -> list[dict]:
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows: raise ValueError(f'Cannot write an empty manifest: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def verify_source(repo: Path) -> dict:
    expected = json.loads((KIT / 'reference_hashes.json').read_text())
    bad = []
    for rel, value in expected.items():
        p = repo / rel
        if not p.is_file(): bad.append(f'MISSING {rel}')
        elif sha256(p) != value: bad.append(f'CHANGED {rel}')
    if bad:
        raise RuntimeError('Audited source differs. Stop rather than guess:\n' + '\n'.join(bad))
    return expected


def load_project(repo: Path, overrides: dict):
    """Import *the supplied code*, after pinning configuration and offline paths."""
    verify_source(repo)
    sys.dont_write_bytecode = True
    for key in list(os.environ):
        if key.startswith(('V921_', 'V13_', 'V15_', 'V15R_', 'V16_', 'V17_', 'V18_')):
            del os.environ[key]
    os.environ.update(HF_HOME=str(repo/'weights'), HF_HUB_CACHE=str(repo/'weights/hub'),
                      HUGGINGFACE_HUB_CACHE=str(repo/'weights/hub'), HF_HUB_OFFLINE='1',
                      TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    sys.path.insert(0, str(repo/'baseline_v20'))
    sys.path.insert(1, str(repo/'EXPERIMENTSRESULT'))
    sys.path.insert(2, str(repo))
    if 'config' in sys.modules:
        raise RuntimeError('Run one experiment per Python process; config already imported.')
    import config
    for key, value in overrides.items(): setattr(config, key, value)
    config.ROOT = repo
    config.DATASET_ROOT = repo
    config.WEIGHTS_DIR = repo/'weights'
    import v9_2_1lib as lib
    import v15_v18_common_train as original
    return config, lib, original


def serializable_config(config) -> dict:
    result = {}
    for name, value in vars(config).items():
        if not name.isupper(): continue
        if isinstance(value, Path): value = str(value)
        try: json.dumps(value)
        except TypeError: continue
        result[name] = value
    return result


def protocol_dir(prepared: Path, dataset: str, target: str) -> Path:
    if dataset == 'original7':
        if target != '77GHz': raise ValueError('Original study is fixed to 10+24 -> 77 GHz.')
        return prepared/'original7'
    if dataset != 'core6' or target not in FREQUENCIES: raise ValueError((dataset, target))
    return prepared/f'core6_to_{target}'


def read_protocol(prepared: Path, dataset: str, target: str, smoke: bool, acknowledge: bool):
    path = protocol_dir(prepared, dataset, target)
    info = json.loads((path/'protocol.json').read_text())
    for name, h in info['manifest_hashes'].items():
        if sha256(path/name) != h:
            raise RuntimeError(f'Manifest changed since preparation: {path/name}')
    if info['missing_or_unreadable_images']:
        raise RuntimeError(f"{len(info['missing_or_unreadable_images'])} missing/unreadable files; read protocol.json")
    if info['duplicate_risk'] and not (smoke or acknowledge):
        raise RuntimeError('Cross-split duplicate risk exists. Read the audit; legacy reproduction requires '
                           '--acknowledge-duplicate-risk. No files are deleted or re-split automatically.')
    return path, info
