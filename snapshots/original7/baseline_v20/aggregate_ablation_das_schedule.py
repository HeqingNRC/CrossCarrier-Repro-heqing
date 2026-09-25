"""Aggregate the V9.2.1 DAS-schedule ablation across 3 seeds.

Compares 4 rungs of the DAS schedule ladder, paired by seed:

    none           : -DAS (no DAS at all)               -> floor
    fixed_narrow   : DAS-as-operator at [10, 30] GHz     -> Kern-style narrow
    fixed_full     : DAS-as-operator at [15, 140] GHz    -> curriculum end-state
    curriculum     : V9.2.1 full 3-stage schedule        -> upper rung

`fixed_full` shares its frequency *range* with the curriculum's final stage
(15-140 GHz). The only difference is the *schedule itself* (1-stage from
ep1 vs. 3-stage warmup -> mid -> wide). This isolates the curriculum's
contribution from the operator's contribution.

Outputs:
    output/V9.2.1-Ablation-DAS-Schedule/aggregate.json
    output/V9.2.1-Ablation-DAS-Schedule/aggregate.md
    paper_metrial/tables/ablation_das_schedule.md
    paper_metrial/data/ablation_das_schedule.json
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path("G:/zhanghe/output")
PAPER = Path("G:/zhanghe/paper_metrial")
OUT_DIR = ROOT / "V9.2.1-Ablation-DAS-Schedule"

SEEDS = [42, 1234, 7890]

# V9.2.1 full (curriculum DAS, DANN off)
FULL_SELF = {
    42:   ROOT / "V9.2.1-KnownPeople-UnknownFreq" / "run_002_strict",
    1234: ROOT / "V9.2.1-KnownPeople-UnknownFreq" / "run_seed1234_strict",
    7890: ROOT / "V9.2.1-KnownPeople-UnknownFreq" / "run_seed7890_strict",
}
FULL_CI4R = {
    42:   ROOT / "V9.2.1-KnownPeople-UnknownFreq-CI4R" / "run_001",
    1234: ROOT / "V9.2.1-KnownPeople-UnknownFreq-CI4R" / "run_seed1234",
    7890: ROOT / "V9.2.1-KnownPeople-UnknownFreq-CI4R" / "run_seed7890",
}

# -DAS (DAS off, DANN off) -- floor
NODAS_SELF = {
    42:   ROOT / "V9.2.1-KnownPeople-UnknownFreq" / "run_no_das_strict",
    1234: ROOT / "V9.2.1-KnownPeople-UnknownFreq" / "run_no_das_seed1234_strict",
    7890: ROOT / "V9.2.1-KnownPeople-UnknownFreq" / "run_no_das_seed7890_strict",
}
NODAS_CI4R = {
    42:   ROOT / "V9.2.1-KnownPeople-UnknownFreq-CI4R" / "run_no_das",
    1234: ROOT / "V9.2.1-KnownPeople-UnknownFreq-CI4R" / "run_no_das_seed1234",
    7890: ROOT / "V9.2.1-KnownPeople-UnknownFreq-CI4R" / "run_no_das_seed7890",
}

# fixed_full (DAS-as-operator at [15,140] GHz, p=1.0, no curriculum)
FIXED_FULL_SELF = {
    s: ROOT / "V9.2.1-KnownPeople-UnknownFreq" / f"run_das_fixed_full_seed{s}_strict"
    for s in SEEDS
}
FIXED_FULL_CI4R = {
    s: ROOT / "V9.2.1-KnownPeople-UnknownFreq-CI4R" / f"run_das_fixed_full_seed{s}"
    for s in SEEDS
}

# fixed_narrow (DAS-as-operator at [10,30] GHz, p=1.0, no curriculum)
FIXED_NARROW_SELF = {
    s: ROOT / "V9.2.1-KnownPeople-UnknownFreq" / f"run_das_fixed_narrow_seed{s}_strict"
    for s in SEEDS
}
FIXED_NARROW_CI4R = {
    s: ROOT / "V9.2.1-KnownPeople-UnknownFreq-CI4R" / f"run_das_fixed_narrow_seed{s}"
    for s in SEEDS
}


def load(rd: Path) -> dict | None:
    p = rd / "reports" / "strict_analysis.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def metrics(s: dict, protocol: str):
    block = "biased_dev77_selection" if protocol == "biased" else "strict_sourceval_selection"
    return s[block]["test77_acc"], s[block]["test77_f1"]


def collect(table, protocol):
    rows = []
    for seed, rd in table.items():
        s = load(rd)
        if s is None:
            continue
        acc, f1 = metrics(s, protocol)
        rows.append({"seed": seed, "acc": acc, "f1": f1})
    return rows


def mean_std(values):
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def fmt_ms(m, s, digits=4):
    if m is None:
        return "--"
    return f"{m:.{digits}f} $\\pm$ {s:.{digits}f}"


def paired_delta(a_rows, b_rows, key):
    """Mean +/- stdev of (a - b) on the seeds present in BOTH lists."""
    a_by = {r["seed"]: r[key] for r in a_rows}
    b_by = {r["seed"]: r[key] for r in b_rows}
    common = sorted(set(a_by) & set(b_by))
    diffs = [a_by[s] - b_by[s] for s in common]
    if not diffs:
        return None, None, []
    if len(diffs) == 1:
        return diffs[0], 0.0, common
    return statistics.mean(diffs), statistics.stdev(diffs), common


def build_panel(label, full_t, fixed_full_t, fixed_narrow_t, nodas_t):
    panel = {"label": label, "rows": {}}
    for protocol in ("biased", "strict"):
        full   = collect(full_t,         protocol)
        ffull  = collect(fixed_full_t,   protocol)
        fnarr  = collect(fixed_narrow_t, protocol)
        nodas  = collect(nodas_t,        protocol)

        def ms(rows, key):
            return mean_std([r[key] for r in rows])

        panel["rows"][protocol] = {
            "curriculum":   {"acc_mean_std": ms(full,  "acc"),
                              "f1_mean_std":  ms(full,  "f1"),
                              "per_seed": full},
            "fixed_full":   {"acc_mean_std": ms(ffull, "acc"),
                              "f1_mean_std":  ms(ffull, "f1"),
                              "per_seed": ffull},
            "fixed_narrow": {"acc_mean_std": ms(fnarr, "acc"),
                              "f1_mean_std":  ms(fnarr, "f1"),
                              "per_seed": fnarr},
            "none":         {"acc_mean_std": ms(nodas, "acc"),
                              "f1_mean_std":  ms(nodas, "f1"),
                              "per_seed": nodas},
            # paired Delta vs curriculum (positive = curriculum wins)
            "delta_vs_curriculum": {
                "fixed_full_f1":   paired_delta(full, ffull, "f1"),
                "fixed_narrow_f1": paired_delta(full, fnarr, "f1"),
                "none_f1":         paired_delta(full, nodas, "f1"),
            },
        }
    return panel


def render_md(panels):
    lines = []
    lines.append("# DAS schedule ablation (V9.2.1)")
    lines.append("")
    lines.append("- Seeds: [42, 1234, 7890]")
    lines.append("- Identical V9.2.1 backbone, training recipe, and selection protocol; "
                 "only the DAS schedule changes.")
    lines.append("  - **None** (-DAS): DAS curriculum disabled (floor).")
    lines.append("  - **Fixed-narrow**: DAS-as-operator at $[10, 30]$ GHz, $p{=}1.0$ "
                 "from epoch 1 (Kern-2022 style, $30/24\\approx1.25{\\times}$ source).")
    lines.append("  - **Fixed-full**: DAS-as-operator at $[15, 140]$ GHz, $p{=}1.0$ from "
                 "epoch 1. Range matches the curriculum's *final stage* exactly; only "
                 "the schedule itself differs.")
    lines.append("  - **Curriculum** (V9.2.1 full): 3-stage schedule (warmup $[10,24]$ "
                 "$p{=}0.35$ -> mid $[10,50]$ $p{=}0.70$ -> wide $[12,95]$ $p{=}1.0$).")
    lines.append("- Selection protocols: `biased` = argmax dev77 F1; `strict` = argmax source-val F1.")
    lines.append("")
    for panel in panels:
        lines.append(f"## {panel['label']}")
        lines.append("")
        lines.append("| DAS schedule | Protocol | acc | F1 |")
        lines.append("|---|---|---|---|")
        for protocol in ("biased", "strict"):
            r = panel["rows"][protocol]
            lab_p = "Biased (target-dev)" if protocol == "biased" else "Strict (source-val)"
            for which, name in (
                ("none",         "None (-DAS)"),
                ("fixed_narrow", "Fixed-narrow [10,30]"),
                ("fixed_full",   "Fixed-full [15,140]"),
                ("curriculum",   "Curriculum (V9.2.1)"),
            ):
                acc_m, acc_s = r[which]["acc_mean_std"]
                f1_m,  f1_s  = r[which]["f1_mean_std"]
                lines.append(
                    f"| {name} | {lab_p} | {fmt_ms(acc_m, acc_s)} | {fmt_ms(f1_m, f1_s)} |"
                )
        lines.append("")
        lines.append(f"### {panel['label']} - paired Delta vs Curriculum (F1, strict, positive = curriculum wins)")
        lines.append("")
        lines.append("| Comparison | mean Delta F1 | stdev | seeds |")
        lines.append("|---|---:|---:|---:|")
        d = panel["rows"]["strict"]["delta_vs_curriculum"]
        for key, name in (
            ("fixed_full_f1",   "Curriculum vs Fixed-full"),
            ("fixed_narrow_f1", "Curriculum vs Fixed-narrow"),
            ("none_f1",         "Curriculum vs None (-DAS)"),
        ):
            m, s, common = d[key]
            if m is None:
                lines.append(f"| {name} | -- | -- | 0 |")
            else:
                lines.append(f"| {name} | {m:+.4f} | {s:.4f} | {len(common)} |")
        lines.append("")
        lines.append(f"### {panel['label']} - per-seed F1, strict protocol")
        lines.append("")
        lines.append("| seed | None | Fixed-narrow | Fixed-full | Curriculum |")
        lines.append("|---|---:|---:|---:|---:|")
        for seed in SEEDS:
            r = panel["rows"]["strict"]
            def find(seed, lst):
                for x in lst:
                    if x["seed"] == seed:
                        return f"{x['f1']:.4f}"
                return "--"
            lines.append(
                f"| {seed} "
                f"| {find(seed, r['none']['per_seed'])} "
                f"| {find(seed, r['fixed_narrow']['per_seed'])} "
                f"| {find(seed, r['fixed_full']['per_seed'])} "
                f"| {find(seed, r['curriculum']['per_seed'])} |"
            )
        lines.append("")
    lines.append("## Reading")
    lines.append("")
    lines.append("- **Curriculum > Fixed-full** isolates the *schedule* itself, since "
                 "Fixed-full uses the same operator and the same final-stage range. The "
                 "remaining gap is the value of the warmup/mid stages.")
    lines.append("- **Fixed-full > None** confirms DAS-as-operator alone (no curriculum) "
                 "is already useful, ruling out the trivial null hypothesis.")
    lines.append("- **Fixed-narrow** shows whether a small extrapolation $[10,30]$ GHz "
                 "(Kern 2022 style) is enough; comparing against Fixed-full isolates the "
                 "value of stretching to far-source frequencies.")
    return "\n".join(lines).strip() + "\n"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panels = [
        build_panel("Self-collected", FULL_SELF, FIXED_FULL_SELF,
                    FIXED_NARROW_SELF, NODAS_SELF),
        build_panel("CI4R-MULTI3",   FULL_CI4R, FIXED_FULL_CI4R,
                    FIXED_NARROW_CI4R, NODAS_CI4R),
    ]

    md = render_md(panels)
    js = json.dumps(panels, indent=2, default=lambda o: list(o) if isinstance(o, tuple) else o)

    (OUT_DIR / "aggregate.md").write_text(md, encoding="utf-8")
    (OUT_DIR / "aggregate.json").write_text(js, encoding="utf-8")

    paper_md = PAPER / "tables" / "ablation_das_schedule.md"
    paper_jsn = PAPER / "data"  / "ablation_das_schedule.json"
    paper_md.parent.mkdir(parents=True, exist_ok=True)
    paper_jsn.parent.mkdir(parents=True, exist_ok=True)
    paper_md.write_text(md, encoding="utf-8")
    paper_jsn.write_text(js, encoding="utf-8")

    print(f"saved {OUT_DIR / 'aggregate.md'}")
    print(f"saved {paper_md}")


if __name__ == "__main__":
    main()
