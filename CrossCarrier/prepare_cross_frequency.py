"""Audit supplied images and prepare reproducible, source-only frequency splits.

This utility uses only the Python standard library. It never copies or changes
the images. Manifest image paths remain relative to the repository root.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

CLASSES = ("Away", "Bend", "Kneel", "Pick", "SStep", "Sit", "Towards")
FREQUENCIES = ("10GHz", "24GHz", "77GHz")
SPLITS = ("train", "val", "test")
FIELDS = ("path", "frequency", "class", "class_idx", "subject", "source_file",
          "original_path", "original_split", "input_manifest_split",
          "original_class_idx", "recording_id", "sha256")
ROOT = Path(__file__).resolve().parent


def recording_key(row: dict) -> tuple[str, str, str]:
    """Carrier-independent capture key, including '(copy)' file variants.

    Filename timestamps identify captures in this supplied dataset. This does
    not establish independence between adjacent captures from one session.
    """
    name = row.get("source_file") or Path(row["path"]).name
    match = re.search(r"_(\d{10})(?!\d)", name)
    if not match:
        raise ValueError(f"Cannot identify recording timestamp: {name}")
    return row["subject"], row["class"], match.group(1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(root: Path, manifest_dir: Path) -> tuple[list[dict], dict]:
    rows, manifests, seen = [], {}, set()
    for split in SPLITS:
        path = manifest_dir / f"{split}.csv"
        manifests[split] = {"path": str(path.relative_to(root)),
                            "sha256": sha256_file(path)}
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for raw in csv.DictReader(handle):
                row = dict(raw)
                if row["frequency"] not in FREQUENCIES:
                    raise ValueError(f"Unexpected frequency: {row['frequency']}")
                image = (root / row["path"]).resolve()
                if not image.is_relative_to(root.resolve()):
                    raise ValueError(f"Image path escapes repository: {row['path']}")
                if not image.is_file():
                    raise FileNotFoundError(image)
                if (image.name != row["source_file"] or image.parent.name != row["class"]
                        or image.parent.parent.name != row["frequency"]):
                    raise ValueError(f"Image path and manifest labels disagree: {row['path']}")
                normalized_path = image.relative_to(root.resolve()).as_posix()
                if normalized_path in seen:
                    raise ValueError(f"Repeated manifest image path: {normalized_path}")
                seen.add(normalized_path)
                row["path"] = normalized_path
                row["input_manifest_split"] = split
                row["original_class_idx"] = row["class_idx"]
                row["recording_id"] = "|".join(recording_key(row))
                row["sha256"] = sha256_file(image)
                rows.append(row)
    return rows, manifests


def counts(rows: list[dict]) -> dict:
    return {"rows": len(rows),
            "frequency": dict(sorted(Counter(r["frequency"] for r in rows).items())),
            "class": dict(sorted(Counter(r["class"] for r in rows).items())),
            "subject": dict(sorted(Counter(r["subject"] for r in rows).items())),
            "frequency_class": dict(sorted(Counter(
                f"{r['frequency']}|{r['class']}" for r in rows).items())),
            "frequency_subject": dict(sorted(Counter(
                f"{r['frequency']}|{r['subject']}" for r in rows).items()))}


def group_overlaps(left: list[dict], right: list[dict], field: str) -> list[str]:
    return sorted({r[field] for r in left} & {r[field] for r in right})


def audit_input(rows: list[dict], manifests: dict) -> dict:
    seven = [r for r in rows if r["class"] in CLASSES]
    hashes = defaultdict(list)
    for row in rows:
        hashes[row["sha256"]].append(row)
    duplicates = [{"sha256": h, "rows": [{k: r[k] for k in
                    ("path", "frequency", "class", "recording_id", "input_manifest_split")}
                    for r in group]}
                  for h, group in sorted(hashes.items()) if len(group) > 1]
    result = {"input_manifests": manifests, "missing_images": 0,
              "repeated_manifest_paths": 0, "all_classes": counts(rows),
              "seven_classes": counts(seven), "identical_file_groups": duplicates,
              "identical_file_group_count": len(duplicates),
              "identical_file_excess_rows": sum(len(g["rows"]) - 1 for g in duplicates),
              "original_frequency_provenance": dict(sorted(Counter(
                  f"{r['frequency']}|{r['original_split']}" for r in rows).items()))}
    for name, subset in (("all_classes", rows), ("seven_classes", seven)):
        train = [r for r in subset if r["input_manifest_split"] == "train"]
        val = [r for r in subset if r["input_manifest_split"] == "val"]
        test = [r for r in subset if r["input_manifest_split"] == "test"]
        result[name]["original_split_counts"] = {s: counts([r for r in subset
                 if r["input_manifest_split"] == s]) for s in SPLITS}
        result[name]["source_train_val_shared_recordings"] = group_overlaps(train, val, "recording_id")
        result[name]["source_train_val_identical_hashes"] = group_overlaps(train, val, "sha256")
        result[name]["source_target_shared_recordings"] = group_overlaps(train + val, test, "recording_id")
    return result


def deduplicate(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Remove exact file copies within a carrier, retaining provenance."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["frequency"], row["sha256"])].append(row)
    retained, removed = [], []
    for _, group in sorted(groups.items()):
        if len({(r["subject"], r["class"]) for r in group}) != 1:
            raise ValueError("Identical files have conflicting subject/class metadata")
        ordered = sorted(group, key=lambda r: ("copy" in r["source_file"].lower(), r["path"]))
        retained.append(ordered[0])
        removed.extend({"removed_path": r["path"], "retained_path": ordered[0]["path"],
                        "sha256": r["sha256"]} for r in ordered[1:])
    return sorted(retained, key=lambda r: r["path"]), removed


def make_group_assignments(rows: list[dict], seed: int, val_fraction: float) -> dict[str, str]:
    """One seeded split assignment per capture/hash component across carriers.

    Identical-file hashes also connect captures so exact copies cannot straddle
    source train/val even when their timestamp metadata differs. Subject/class
    strata with fewer than five components remain in training, matching the
    original small-stratum policy. The ranking is independent of input order.
    """
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between zero and one")
    parent = {row["recording_id"]: row["recording_id"] for row in rows}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    hashes = defaultdict(list)
    for row in rows:
        hashes[row["sha256"]].append(row)
    for group in hashes.values():
        first = find(group[0]["recording_id"])
        for row in group[1:]:
            other = find(row["recording_id"])
            lo, hi = sorted((find(first), other))
            parent[hi] = lo
            first = lo
    components = defaultdict(list)
    for row in rows:
        components[find(row["recording_id"])].append(row)
    strata = defaultdict(set)
    for key, component in components.items():
        labels = {(row["subject"], row["class"]) for row in component}
        if len(labels) != 1:
            raise ValueError("Identical-image connected recordings have conflicting subject/class metadata")
        strata[next(iter(labels))].add(key)
    assignments = {}
    for _, keys in sorted(strata.items()):
        ranked = sorted(keys, key=lambda key: (
            hashlib.sha256(f"{seed}|{key}".encode()).hexdigest(), key))
        nval = min(len(ranked) - 1, max(1, int(len(ranked) * val_fraction + 0.5))) if len(ranked) >= 5 else 0
        val_keys = set(ranked[:nval])
        for key in ranked:
            split = "val" if key in val_keys else "train"
            assignments.update({row["recording_id"]: split for row in components[key]})
    return assignments


def build_grouped(rows: list[dict], target: str, assignments: dict) -> dict[str, list[dict]]:
    splits = {s: [] for s in SPLITS}
    for row in rows:
        split = "test" if row["frequency"] == target else assignments[row["recording_id"]]
        splits[split].append(row)
    return splits


def validate_direction(splits: dict, target: str, *, strict: bool) -> dict:
    train, val, test = (splits[s] for s in SPLITS)
    source = train + val
    if not all(splits.values()):
        raise ValueError("A generated split is empty")
    if any(r["frequency"] == target for r in source):
        raise ValueError("Target frequency leaked into source train/val")
    if any(r["frequency"] != target for r in test):
        raise ValueError("Test contains a source frequency")
    source_freqs = set(FREQUENCIES) - {target}
    if {r["frequency"] for r in source} != source_freqs:
        raise ValueError("Source frequency missing")
    if not {r["subject"] for r in test}.issubset({r["subject"] for r in train}):
        raise ValueError("A target subject is absent from source training")
    for split, split_rows in splits.items():
        if {r["class"] for r in split_rows} != set(CLASSES):
            raise ValueError(f"Class coverage is incomplete in {split}")
        for frequency in {r["frequency"] for r in split_rows}:
            if {r["class"] for r in split_rows if r["frequency"] == frequency} != set(CLASSES):
                raise ValueError(f"Class coverage is incomplete in {split}/{frequency}")
    path_overlaps = {f"{a}_{b}": group_overlaps(splits[a], splits[b], "path")
                     for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    if any(path_overlaps.values()):
        raise ValueError("Image paths overlap across splits")
    recording_overlap = group_overlaps(train, val, "recording_id")
    hash_overlap = group_overlaps(train, val, "sha256")
    if strict and (recording_overlap or hash_overlap):
        raise ValueError("Source train/val recording or identical-image leakage")
    if group_overlaps(source, test, "sha256"):
        raise ValueError("Identical target image in source train/val")
    return {"target_frequency": target, "source_frequencies": sorted(source_freqs),
            "split_counts": {s: counts(rs) for s, rs in splits.items()},
            "path_overlaps": path_overlaps,
            "source_train_val_shared_recordings": recording_overlap,
            "source_train_val_identical_hashes": hash_overlap,
            "source_target_shared_recordings": group_overlaps(source, test, "recording_id"),
            "target_training_shared_recordings": group_overlaps(train, test, "recording_id"),
            "recording_disjoint_source_validation": not recording_overlap,
            "notes": ["Target labels must never select checkpoints or hyperparameters.",
                      "Source and target may contain synchronized captures of the same recording.",
                      "This evaluates known-person cross-frequency transfer, not unseen recordings or sessions."]}


def write_direction(root: Path, relative_dir: str, splits: dict, report: dict) -> None:
    task_dir = root / relative_dir
    manifest_dir = task_dir / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        with (manifest_dir / f"{split}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    (task_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--protocol", choices=("both", "legacy", "grouped"), default="grouped")
    args = parser.parse_args()
    root = args.root.resolve()
    rows, manifests = load_rows(root, root / "tasks/known_people_unknown_freq/manifest")
    audit = audit_input(rows, manifests)
    rows = [dict(r, class_idx=str(CLASSES.index(r["class"]))) for r in rows if r["class"] in CLASSES]
    audit["generator"] = {"seed": args.seed, "val_fraction": args.val_fraction,
                          "class_order": list(CLASSES), "protocol": args.protocol}
    audit["generated_directions"] = {}
    if args.protocol in ("both", "legacy"):
        splits = {s: [r for r in rows if r["input_manifest_split"] == s] for s in SPLITS}
        report = validate_direction(splits, "77GHz", strict=False)
        report["protocol"] = "legacy77: preserve provided split membership and duplicate weighting; seven classes"
        report["warning"] = "Original source train/val contains shared recordings and identical-image copies. Reproduction only."
        directory = "tasks/cross_frequency/target77"
        write_direction(root, directory, splits, report)
        audit["generated_directions"][directory] = report
    if args.protocol in ("both", "grouped"):
        unique, removed = deduplicate(rows)
        assignments = make_group_assignments(rows, args.seed, args.val_fraction)
        audit["grouped_deduplicated_rows"] = counts(unique)
        audit["grouped_removed_duplicate_paths"] = removed
        audit["grouped_recording_assignments"] = assignments
        for target in FREQUENCIES:
            splits = build_grouped(unique, target, assignments)
            report = validate_direction(splits, target, strict=True)
            report["protocol"] = "grouped7: exact-file deduplication; global subject/class/timestamp and hash-component split"
            report["split_seed"] = args.seed
            report["requested_val_fraction"] = args.val_fraction
            directory = f"tasks/cross_frequency_grouped/target{target.removesuffix('GHz')}"
            write_direction(root, directory, splits, report)
            audit["generated_directions"][directory] = report
    audit_path = root / "tasks/cross_frequency_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    for directory, report in audit["generated_directions"].items():
        print(f"{directory}: " + ", ".join(f"{s}={c['rows']}" for s, c in report["split_counts"].items()))
    print(f"Audit written: {audit_path}")


if __name__ == "__main__":
    main()
