"""Aggregate biased + strict V9.2.1 numbers across multiple seeds.

Reads `strict_analysis.json` from every run-dir matching the V9.2.1
self / CI4R seed-suffix patterns and emits mean +/- std for each
(dataset, protocol) combination.

Outputs:
    output/V9.2.1-MultiSeed/aggregate.json
    output/V9.2.1-MultiSeed/aggregate.md
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path("G:/zhanghe/output")
SELF_RUNS = [
    ROOT / "V9.2.1-KnownPeople-UnknownFreq" / "run_002_strict",          # seed 42
    ROOT / "V9.2.1-KnownPeople-UnknownFreq" / "run_seed1234_strict",
    ROOT / "V9.2.1-KnownPeople-UnknownFreq" / "run_seed7890_strict",
]
CI4R_RUNS = [
    ROOT / "V9.2.1-KnownPeople-UnknownFreq-CI4R" / "run_001",            # seed 42
    ROOT / "V9.2.1-KnownPeople-UnknownFreq-CI4R" / "run_seed1234",
    ROOT / "V9.2.1-KnownPeople-UnknownFreq-CI4R" / "run_seed7890",
]
OUT_DIR = ROOT / "V9.2.1-MultiSeed"


def load_strict_json(run_dir: Path):
    p = run_dir / "reports" / "strict_analysis.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def collect(runs):
    rows = []
    for rd in runs:
        s = load_strict_json(rd)
        if s is None:
            print(f"  ! missing: {rd}/reports/strict_analysis.json")
            continue
        rows.append({
            "run_dir": str(rd),
            "biased_acc":  s["biased_dev77_selection"]["test77_acc"],
            "biased_f1":   s["biased_dev77_selection"]["test77_f1"],
            "biased_ep":   s["biased_dev77_selection"]["best_epoch"],
            "strict_acc":  s["strict_sourceval_selection"]["test77_acc"],
            "strict_f1":   s["strict_sourceval_selection"]["test77_f1"],
            "strict_ep":   s["strict_sourceval_selection"]["best_epoch"],
        })
    return rows


def fmt_mean_std(values):
    if not values:
        return "--", "--"
    if len(values) == 1:
        return f"{values[0]:.4f}", f"{0.0:.4f}"
    return f"{statistics.mean(values):.4f}", f"{statistics.stdev(values):.4f}"


def make_summary(label, rows):
    biased_acc = [r["biased_acc"] for r in rows if r["biased_acc"] is not None]
    biased_f1  = [r["biased_f1"]  for r in rows if r["biased_f1"]  is not None]
    strict_acc = [r["strict_acc"] for r in rows if r["strict_acc"] is not None]
    strict_f1  = [r["strict_f1"]  for r in rows if r["strict_f1"]  is not None]

    return {
        "label": label,
        "n_seeds": len(rows),
        "runs": [str(r["run_dir"]) for r in rows],
        "biased_acc_mean": (statistics.mean(biased_acc) if biased_acc else None),
        "biased_acc_std":  (statistics.stdev(biased_acc) if len(biased_acc) > 1 else 0.0),
        "biased_f1_mean":  (statistics.mean(biased_f1)  if biased_f1  else None),
        "biased_f1_std":   (statistics.stdev(biased_f1)  if len(biased_f1)  > 1 else 0.0),
        "strict_acc_mean": (statistics.mean(strict_acc) if strict_acc else None),
        "strict_acc_std":  (statistics.stdev(strict_acc) if len(strict_acc) > 1 else 0.0),
        "strict_f1_mean":  (statistics.mean(strict_f1)  if strict_f1  else None),
        "strict_f1_std":   (statistics.stdev(strict_f1)  if len(strict_f1)  > 1 else 0.0),
        "per_run":         rows,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    self_rows = collect(SELF_RUNS)
    ci4r_rows = collect(CI4R_RUNS)
    self_summary = make_summary("self-collected", self_rows)
    ci4r_summary = make_summary("CI4R-MULTI3",    ci4r_rows)

    overall = {"self": self_summary, "ci4r": ci4r_summary}
    (OUT_DIR / "aggregate.json").write_text(
        json.dumps(overall, indent=2), encoding="utf-8"
    )

    md = ["# V9.2.1 multi-seed aggregate", "",
          f"- Generated: {Path(__file__).stat().st_mtime}", ""]
    for label_key, summ in (("self", self_summary), ("ci4r", ci4r_summary)):
        md.append(f"## {summ['label']} (n={summ['n_seeds']} seeds)")
        md.append("")
        md.append("| Protocol | acc (mean +/- std) | F1 (mean +/- std) |")
        md.append("|---|---|---|")
        md.append(
            f"| Biased (target-dev) "
            f"| {summ['biased_acc_mean']:.4f} +/- {summ['biased_acc_std']:.4f} "
            f"| {summ['biased_f1_mean']:.4f} +/- {summ['biased_f1_std']:.4f} |"
        )
        md.append(
            f"| Strict (source-val) "
            f"| {summ['strict_acc_mean']:.4f} +/- {summ['strict_acc_std']:.4f} "
            f"| {summ['strict_f1_mean']:.4f} +/- {summ['strict_f1_std']:.4f} |"
        )
        md.append("")
        md.append("### per-seed")
        md.append("")
        md.append("| run_dir | biased acc | biased F1 | strict acc | strict F1 |")
        md.append("|---|---:|---:|---:|---:|")
        for r in summ["per_run"]:
            md.append(
                f"| `{Path(r['run_dir']).name}` "
                f"| {r['biased_acc']:.4f} | {r['biased_f1']:.4f} "
                f"| {r['strict_acc']:.4f} | {r['strict_f1']:.4f} |"
            )
        md.append("")

    (OUT_DIR / "aggregate.md").write_text("\n".join(md).strip() + "\n", encoding="utf-8")
    print(f"saved {OUT_DIR/'aggregate.json'}")
    print(f"saved {OUT_DIR/'aggregate.md'}")
    print()
    for label_key, summ in (("self", self_summary), ("ci4r", ci4r_summary)):
        print(f"  [{summ['label']}] n={summ['n_seeds']}")
        if summ['biased_f1_mean'] is not None:
            print(f"    biased: acc {summ['biased_acc_mean']:.4f}+/-{summ['biased_acc_std']:.4f}  "
                  f"F1 {summ['biased_f1_mean']:.4f}+/-{summ['biased_f1_std']:.4f}")
        if summ['strict_f1_mean'] is not None:
            print(f"    strict: acc {summ['strict_acc_mean']:.4f}+/-{summ['strict_acc_std']:.4f}  "
                  f"F1 {summ['strict_f1_mean']:.4f}+/-{summ['strict_f1_std']:.4f}")


if __name__ == "__main__":
    main()
