#!/usr/bin/env python3
"""Read-only image/split audit. Creates NEW protocol files; never changes datasets.
Run on Sockeye. No GPU work, model imports, pip installs, or checkpoint unpickling.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, re, sys
from pathlib import Path
from PIL import Image
from xccontrol.common import *


def collect_rows(repo: Path, dataset: str, core_root: Path | None):
    rows = []
    if dataset == 'original7':
        for split in ('train', 'val', 'test'):
            source = repo/f'tasks/known_people_unknown_freq/manifest/{split}.csv'
            # Identity check: refuse a different original manifest without re-auditing it.
            if sha256(source) != sha256(KIT/f'manifests/original7/{split}.csv'):
                raise RuntimeError(f'Original manifest changed: {source}')
            for i, r in enumerate(read_csv(source)):
                if r['class'] not in CLASSES7: continue
                p = (repo/r['path'].replace('\\', '/')).resolve()
                rows.append(dict(path=str(p), frequency=r['frequency'], **{'class': r['class']},
                                 split=split, subject=r.get('subject',''),
                                 provenance_manifest=str(source), provenance_row=i+2))
    else:
        for f in FREQUENCIES:
            source = KIT/f'manifests/core6_source/{f}.csv'
            for i, r in enumerate(read_csv(source)):
                if r['class_name'] not in CLASSES6: raise ValueError(r)
                if int(r['class_id']) != CLASSES6.index(r['class_name']): raise ValueError('Label mapping changed')
                p = Path(r['source_file'])
                if core_root is not None:
                    # Preserve the exact frequency/class/basename suffix; no fuzzy matching.
                    p = core_root/f/r['class_name']/p.name
                rows.append(dict(path=str(p.resolve()), frequency=f, **{'class': r['class_name']},
                                 split=r['split'], subject='', provenance_manifest=str(source), provenance_row=i+2))
    return rows


def fingerprint(path: str) -> dict:
    p = Path(path)
    result = dict(path=path, exists=p.is_file())
    if not p.is_file(): return {**result, 'error':'missing'}
    try:
        result['file_sha256'] = sha256(p)
        with Image.open(p) as im:
            result.update(width=im.width, height=im.height, image_mode=im.mode)
            rgb = im.convert('RGB'); rgb.load()
            # Include dimensions: the same byte stream with a different shape is not identical input.
            result['rgb_sha256'] = hashlib.sha256(
                f'{rgb.width}x{rgb.height}:RGB:'.encode()+rgb.tobytes()).hexdigest()
            # Also inspect the actual 224x224 RGB input before augmentations/normalization.
            small = rgb.resize((224,224), Image.Resampling.BILINEAR)
            result['resized224_sha256'] = hashlib.sha256(small.tobytes()).hexdigest()
    except Exception as e:
        result['error'] = f'{type(e).__name__}: {e}'
    return result


def cross_split_groups(rows, fingerprints, field):
    grouped = collections.defaultdict(list)
    for r in rows:
        h = fingerprints[r['path']].get(field)
        if h: grouped[h].append({k:r[k] for k in ('path','frequency','class','split')})
    return [dict(digest=h, rows=rs) for h,rs in grouped.items()
            if len({r['split'] for r in rs}) > 1]


def name_candidates(rows):
    groups = collections.defaultdict(list)
    for r in rows:
        name = re.sub(r' \((?:another )?copy(?: \d+)?\)', '', Path(r['path']).name, flags=re.I)
        groups[(r['frequency'],r['class'],name)].append(r)
    return [dict(key=list(k), rows=[{v:r[v] for v in ('path','split')} for r in rs])
            for k,rs in groups.items() if len({r['split'] for r in rs}) > 1]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', type=Path, default=DEFAULT_REPO)
    ap.add_argument('--out', type=Path, default=KIT/'prepared')
    ap.add_argument('--core-root', type=Path, help='Only if the original source image root moved')
    args=ap.parse_args(); repo=args.repo.resolve(); out=args.out.resolve()
    if out.exists(): raise FileExistsError(f'Refusing to overwrite {out}; use a new --out path.')
    source_hashes=verify_source(repo)
    rows_by_dataset={d:collect_rows(repo,d,args.core_root) for d in ('original7','core6')}
    out.mkdir(parents=True)
    cache={}
    all_paths=sorted({r['path'] for rs in rows_by_dataset.values() for r in rs})
    for i,path in enumerate(all_paths):
        cache[path]=fingerprint(path)
        if (i+1)%200==0: print(f'[hash] {i+1}/{len(all_paths)}', flush=True)
    write_json(out/'image_fingerprints.json', cache)
    audits={}
    for dataset, rows in rows_by_dataset.items():
        audit=dict(file_sha256_cross_split=cross_split_groups(rows,cache,'file_sha256'),
                   rgb_sha256_cross_split=cross_split_groups(rows,cache,'rgb_sha256'),
                   resized224_cross_split=cross_split_groups(rows,cache,'resized224_sha256'),
                   filename_candidates=name_candidates(rows),
                   note='Candidate filenames are not proof. Hash groups verify identical bytes/RGB/input. No deduplication is applied.')
        audits[dataset]=audit
        targets=['77GHz'] if dataset=='original7' else FREQUENCIES
        for target in targets:
            classes=CLASSES7 if dataset=='original7' else CLASSES6
            sources=[f for f in FREQUENCIES if f!=target]
            chosen=[]
            for split in ('train','val','test'):
                chosen += [r for r in rows if r['split']==split and
                           ((split=='test' and r['frequency']==target) or
                            (split!='test' and r['frequency'] in sources))]
            folder=out/(dataset if dataset=='original7' else f'core6_to_{target}')
            folder.mkdir()
            for split in ('train','val','test'):
                selected=[dict(r, class_idx_7c=classes.index(r['class']),
                               freq_idx=sources.index(r['frequency']) if r['frequency'] in sources else -1)
                          for r in chosen if r['split']==split]
                write_csv(folder/f'{split}.csv',selected)
            used_duplicates=cross_split_groups(chosen,cache,'resized224_sha256')
            missing=[cache[r['path']] for r in chosen if cache[r['path']].get('error')]
            counts={s:dict(collections.Counter(r['frequency'] for r in chosen if r['split']==s))
                    for s in ('train','val','test')}
            info=dict(dataset=dataset, target=target, sources=sources, classes=classes,
                      protocol='original7_fixed_full418' if dataset=='original7' else 'core6_legacy_image_split_70_10_20',
                      counts=counts, n={s:sum(counts[s].values()) for s in counts},
                      source_repo=str(repo), source_hashes=source_hashes,
                      manifest_hashes={f'{s}.csv':sha256(folder/f'{s}.csv') for s in ('train','val','test')},
                      missing_or_unreadable_images=missing,
                      duplicate_risk=bool(used_duplicates), active_cross_split_identical_input=used_duplicates,
                      axis_calibration_verified=False, subject_disjoint_verified=False,
                      caveats=['No target image is used for training or checkpoint selection by train_controlled.py.',
                               'Class/frequency counts do not establish participant independence or common Hz/pixel.',
                               'Original images/splits preserved; duplicates, if any, remain explicitly flagged.',
                               'Core6 HDF5 preprocessing equivalence to earlier AIRHAR input remains unverified.'])
            write_json(folder/'protocol.json',info)
            print(f"[protocol] {folder.name}: {info['n']}; active duplicate groups={len(used_duplicates)}; missing={len(missing)}")
        print(f"[audit] {dataset}: filename candidates={len(audit['filename_candidates'])}; RGB cross-split groups={len(audit['rgb_sha256_cross_split'])}")
    write_json(out/'duplicates.json',audits)
    weights=list((repo/'weights/hub').glob('models--timm--vit_large_patch16_dinov3.lvd1689m/snapshots/*/model.safetensors'))
    write_json(out/'summary.json',dict(
        repo=str(repo), prepared=str(out), unique_image_paths=len(cache),
        unreadable=sum(bool(r.get('error')) for r in cache.values()),
        backbone_files=[dict(path=str(p),bytes=p.stat().st_size) for p in weights],
        notes='Backbone existence/size is checked, not cryptographic integrity. Training initializes from this local cache; final checkpoints are not used to initialize new runs.'))
    print('[done] PREPARATION COMPLETE. Read prepared/summary.json and duplicates.json. No training was run.')

if __name__=='__main__':
    try: main()
    except Exception as e:
        print(f'[FATAL] {type(e).__name__}: {e}',file=sys.stderr); raise
