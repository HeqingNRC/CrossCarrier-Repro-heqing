#!/usr/bin/env python3
"""Restore the recorded Core6 image splits without a recursive filesystem search.
Only the exact paths in the three included split manifests are inspected.
"""
from __future__ import annotations
import argparse, collections, json, os, shutil, sys, uuid
from pathlib import Path
from xccontrol.common import KIT, CLASSES6, FREQUENCIES, sha256, write_csv, write_json, verify_source
from prepare import collect_rows, fingerprint, cross_split_groups, name_candidates
from suite_constants import SPLIT_COUNTS, EXPECTED_N


def prepare_core6(repo: Path, out: Path, core_root: Path | None, preserve: bool):
    if out.exists():
        raise FileExistsError(f'Prepared data already exists: {out}; it is not overwritten.')
    source_hashes = verify_source(repo)
    rows = collect_rows(repo, 'core6', core_root)
    counts = {f:dict(collections.Counter(r['split'] for r in rows if r['frequency']==f)) for f in FREQUENCIES}
    if counts != SPLIT_COUNTS or len(rows) != 1336:
        raise RuntimeError(f'Historical split counts changed: {counts}')
    if len({r['path'] for r in rows}) != len(rows):
        raise RuntimeError('A raw path occurs more than once in the supplied split manifests')
    missing = [r['path'] for r in rows if not Path(r['path']).is_file()]
    if missing:
        write_json(KIT/'preparation_failure.json',dict(status='MISSING_IMAGES',count=len(missing),paths=missing,
            explanation='Only exact manifest paths were checked; no filesystem-wide search was attempted.',
            action='Restore the recorded image tree, or rerun prepare with --core-root PATH pointing to 10GHz/24GHz/77GHz.'))
        raise FileNotFoundError(f'{len(missing)} Core6 images not found. First: {missing[:3]}. Full list: preparation_failure.json')
    stage = out.parent / (out.name+'.staging-'+uuid.uuid4().hex[:8])
    stage.mkdir(parents=True,exist_ok=False)
    cache = {}
    for i,r in enumerate(rows):
        cache[r['path']] = fingerprint(r['path'])
        if (i+1)%100 == 0 or i+1 == len(rows):
            print(f'[Core6 audit] {i+1}/{len(rows)} exact listed PNG files',flush=True)
    unreadable = [v for v in cache.values() if v.get('error')]
    write_json(stage/'image_fingerprints.json',cache)
    if unreadable:
        write_json(KIT/'preparation_failure.json',dict(status='UNREADABLE_IMAGES',images=unreadable,stage=str(stage)))
        raise RuntimeError(f'{len(unreadable)} unreadable images; see preparation_failure.json')
    duplicate_summary = {}; protocol_list = []
    for target in FREQUENCIES:
        sources = [f for f in FREQUENCIES if f != target]
        chosen = [r for r in rows if (r['split']=='test' and r['frequency']==target) or
                  (r['split'] in ('train','val') and r['frequency'] in sources)]
        identical = cross_split_groups(chosen,cache,'resized224_sha256')
        conflicts = [g for g in identical if len({r['class'] for r in g['rows']})>1]
        if conflicts:
            write_json(KIT/'preparation_failure.json',dict(status='CROSS_LABEL_IDENTICAL_INPUTS',target=target,conflicts=conflicts))
            raise RuntimeError('Identical active input has conflicting class labels; no training submitted')
        if identical and not preserve:
            write_json(KIT/'preparation_failure.json',dict(status='LEGACY_DUPLICATE_ACK_REQUIRED',target=target,groups=identical))
            raise RuntimeError('Historical split contains duplicate inputs. Use --preserve-legacy-splits to keep it explicitly.')
        folder = stage/f'core6_to_{target}'
        folder.mkdir()
        for split in ('train','val','test'):
            selected = [dict(r, class_idx_7c=CLASSES6.index(r['class']),
                            freq_idx=sources.index(r['frequency']) if r['frequency'] in sources else -1,
                            file_sha256=cache[r['path']]['file_sha256']) for r in chosen if r['split']==split]
            if len(selected) != EXPECTED_N[target][split]:
                raise RuntimeError(f'Unexpected {target}/{split} count')
            write_csv(folder/f'{split}.csv',selected)
        info = dict(dataset='core6',target=target,sources=sources,classes=CLASSES6,
                    protocol='core6_legacy_image_split_70_10_20',n=EXPECTED_N[target],
                    source_repo=str(repo),source_hashes=source_hashes,
                    manifest_hashes={f'{s}.csv':sha256(folder/f'{s}.csv') for s in ('train','val','test')},
                    source_split_hashes={f:sha256(KIT/f'manifests/core6_source/{f}.csv') for f in FREQUENCIES},
                    missing_or_unreadable_images=[],duplicate_risk=bool(identical),
                    active_cross_split_identical_input=identical,
                    legacy_preservation_acknowledged=preserve,
                    axis_calibration_verified=False,subject_disjoint_verified=False,
                    hdf5_tensor_equivalence_verified=False,
                    note='Historical PNG paths and image-level splits retained. No dataset file is moved or modified. Do not claim subject/session independence.')
        write_json(folder/'protocol.json',info)
        duplicate_summary[target]=identical
        protocol_list.append(dict(target=target,n=EXPECTED_N[target],duplicate_groups=len(identical)))
        print(f'[Core6 protocol] {target}: {EXPECTED_N[target]}; cross-split duplicate groups={len(identical)}',flush=True)
    write_json(stage/'duplicates.json',duplicate_summary)
    summary = dict(status='CORE6_PREPARATION_PASS',raw_images=1336,classes=CLASSES6,
                   source_root_override=str(core_root) if core_root else None,
                   protocols=protocol_list,unreadable=0,preserve_legacy_splits=preserve,
                   no_training_submitted=True)
    write_json(stage/'summary.json',summary)
    stage.rename(out)
    print('CORE6_DATA_READY. Original files unchanged.',flush=True)
    return summary

if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo',type=Path,default=KIT/'evidence')
    ap.add_argument('--out',type=Path,default=KIT/'prepared')
    ap.add_argument('--core-root',type=Path)
    ap.add_argument('--preserve-legacy-splits',action='store_true')
    a=ap.parse_args()
    prepare_core6(a.repo.resolve(),a.out.resolve(),a.core_root.resolve() if a.core_root else None,a.preserve_legacy_splits)
