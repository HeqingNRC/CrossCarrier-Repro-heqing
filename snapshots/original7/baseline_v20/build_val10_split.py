"""Build the NEW-protocol val/test split for the 77GHz target set.

New protocol (2026-06-08):
  - Carve ~1/10 of the 77GHz TEST set out as a target-domain validation set.
  - Checkpoint selection will later pick the weight with the highest val
    macro-F1; the remaining 90% is the final held-out test set.

This reuses the repo's EXISTING split philosophy from v9_2_1lib:
  - per-class stratification (every class represented in val so val macro-F1
    is well defined),
  - subject-aware *proportional* quotas inside each class
    (`_subject_dev_quotas`), and
  - deterministic, seeded selection via a stable SHA1 key
    (`_stable_split_key`),
the only change being that the per-class val size is round(FRAC * n_class)
instead of a fixed count.

Outputs (written next to the original manifest):
  - test77_val_p10_seed42.csv         (the new validation set, ~10%)
  - test77_finaltest_p10_seed42.csv   (the new held-out test set, ~90%)
  - test77_val10_split_info.json      (full split bookkeeping)
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Config (kept local so this script has no torch / GPU import side effects).
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent          # ...\Letter journal
TASK_DIR = ROOT / "tasks" / "known_people_unknown_freq"
MANIFEST_DIR = TASK_DIR / "manifest"
TEST_CSV = MANIFEST_DIR / "test.csv"

CLASSES = ["Away", "Bend", "Kneel", "Pick", "SStep", "Sit", "Towards"]
VAL_FRAC = 0.10
SEED = 42

VAL_OUT = MANIFEST_DIR / "test77_val_p10_seed42.csv"
TEST_OUT = MANIFEST_DIR / "test77_finaltest_p10_seed42.csv"
INFO_OUT = MANIFEST_DIR / "test77_val10_split_info.json"


# ---------------------------------------------------------------------------
# Helpers copied verbatim from v9_2_1lib (pure functions, no deps).
# ---------------------------------------------------------------------------
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
            remainders, key=lambda x: (-x[2], -(x[1]), x[0]),
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


def _val_count_for_class(n_cls: int, frac: float) -> int:
    """round(frac * n) but always >=1 and <= n-1 (val must be a proper subset)."""
    raw = int(round(frac * n_cls))
    raw = max(1, raw)
    raw = min(raw, n_cls - 1)
    return raw


# ---------------------------------------------------------------------------
# Build the split.
# ---------------------------------------------------------------------------
def main() -> None:
    df = pd.read_csv(TEST_CSV)
    df = df[df["class"].isin(CLASSES)].reset_index(drop=True).copy()

    assert (df["frequency"] == "77GHz").all(), "test manifest must be all 77GHz"

    val_parts, test_parts = [], []
    info = {"protocol": "val10_target_F1_selection", "frac": VAL_FRAC,
            "seed": SEED, "classes": {}}

    for cls in CLASSES:
        cls_df = df[df["class"] == cls].copy()
        n_cls = len(cls_df)
        val_n = _val_count_for_class(n_cls, VAL_FRAC)

        counts = {
            str(s): int(n)
            for s, n in cls_df.groupby("subject").size().to_dict().items()
        }
        quotas = _subject_dev_quotas(counts, val_n)

        for subject in sorted(counts):
            subj_df = cls_df[cls_df["subject"] == subject].copy()
            subj_df["_k"] = subj_df["path"].map(
                lambda p: _stable_split_key(SEED, cls, subject, str(p))
            )
            subj_df = subj_df.sort_values(["_k", "path"], kind="mergesort").reset_index(drop=True)
            k = int(quotas.get(subject, 0))
            val_parts.append(subj_df.iloc[:k].drop(columns=["_k"]))
            test_parts.append(subj_df.iloc[k:].drop(columns=["_k"]))

        info["classes"][cls] = {
            "total": int(n_cls),
            "val": int(val_n),
            "test": int(n_cls - val_n),
            "val_frac_actual": round(val_n / n_cls, 4),
            "subject_counts": counts,
            "subject_val_quotas": {str(k): int(v) for k, v in quotas.items()},
        }

    val_df = pd.concat(val_parts, ignore_index=True).sort_values(
        ["class", "subject", "path"], kind="mergesort").reset_index(drop=True)
    test_df = pd.concat(test_parts, ignore_index=True).sort_values(
        ["class", "subject", "path"], kind="mergesort").reset_index(drop=True)

    # Integrity checks.
    assert set(val_df["path"]).isdisjoint(set(test_df["path"])), "val/test overlap!"
    assert len(val_df) + len(test_df) == len(df), "row count mismatch!"

    val_df.to_csv(VAL_OUT, index=False)
    test_df.to_csv(TEST_OUT, index=False)
    info["totals"] = {
        "test77_full": int(len(df)),
        "val": int(len(val_df)),
        "final_test": int(len(test_df)),
        "val_frac_actual": round(len(val_df) / len(df), 4),
    }
    INFO_OUT.write_text(json.dumps(info, indent=2), encoding="utf-8")

    # ---- Report ----
    def dist(frame):
        return frame.groupby("class").size().reindex(CLASSES).astype(int)

    full_d, val_d, test_d = dist(df), dist(val_df), dist(test_df)

    print("=" * 70)
    print("77GHz TARGET SPLIT  (new protocol: val-F1 selection)")
    print(f"  seed={SEED}  frac={VAL_FRAC}  -> per-class stratified, subject-aware")
    print("=" * 70)
    print(f"{'class':<10}{'full':>6}{'val':>6}{'test':>6}{'val%':>8}")
    print("-" * 70)
    for c in CLASSES:
        print(f"{c:<10}{full_d[c]:>6}{val_d[c]:>6}{test_d[c]:>6}"
              f"{100*val_d[c]/full_d[c]:>7.1f}%")
    print("-" * 70)
    print(f"{'TOTAL':<10}{len(df):>6}{len(val_df):>6}{len(test_df):>6}"
          f"{100*len(val_df)/len(df):>7.1f}%")
    print("=" * 70)

    print("\nPer-class subject breakdown (full -> val):")
    subjects = sorted(df["subject"].unique())
    header = f"{'class':<10}" + "".join(f"{s:>12}" for s in subjects)
    print(header)
    for c in CLASSES:
        row = f"{c:<10}"
        for s in subjects:
            full_cs = len(df[(df["class"] == c) & (df["subject"] == s)])
            val_cs = len(val_df[(val_df["class"] == c) & (val_df["subject"] == s)])
            row += f"{str(full_cs)+'->'+str(val_cs):>12}"
        print(row)

    print("\nSubject totals (full / val / test):")
    for s in subjects:
        print(f"  {s}:  full={len(df[df['subject']==s]):>4}  "
              f"val={len(val_df[val_df['subject']==s]):>4}  "
              f"test={len(test_df[test_df['subject']==s]):>4}")

    print(f"\nWrote:\n  {VAL_OUT}\n  {TEST_OUT}\n  {INFO_OUT}")


if __name__ == "__main__":
    main()
