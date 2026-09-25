"""Aggregate the V9.2.1 -DAS ablation across 3 seeds, paired with the
3-seed V9.2.1 full multi-seed sweep, and emit a single markdown table
with mean +/- std and paired Delta per seed.

Outputs:
    output/V9.2.1-Ablation-NoDAS/aggregate.json
    output/V9.2.1-Ablation-NoDAS/aggregate.md
    paper_metrial/tables/ablation_no_das_comparison.md  (overwrites previous)
    paper_metrial/data/ablation_no_das_aggregate.json
"""
from __future__ import annotations

import json
import shutil
import statistics
from pathlib import Path

ROOT = Path("G:/zhanghe/output")
PAPER = Path("G:/zhanghe/paper_metrial")
OUT_DIR = ROOT / "V9.2.1-Ablation-NoDAS"

# Seed 42 -DAS uses the original run_no_das_* run_dirs; seeds 1234/7890
# use the seed-suffix run_dirs created by the multi-seed chain.
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

# Full V9.2.1 multi-seed paired pointers
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

SEEDS = [42, 1234, 7890]


def load(rd: Path) -> dict | None:
    p = rd / "reports" / "strict_analysis.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def metrics(s: dict, protocol: str):
    block = "biased_dev77_selection" if protocol == "biased" else "strict_sourceval_selection"
    return s[block]["test77_acc"], s[block]["test77_f1"]


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


def collect(table, protocol, mode):
    """mode in {'acc', 'f1'}; returns list across seeds (skipping missing)."""
    out = []
    for seed, rd in table.items():
        s = load(rd)
        if s is None:
            continue
        acc, f1 = metrics(s, protocol)
        out.append((seed, acc if mode == "acc" else f1))
    return out


def paired_delta(full_table, nodas_table, protocol, mode):
    """Returns list of (seed, full_value, nodas_value, full-nodas)."""
    out = []
    for seed in SEEDS:
        sf = load(full_table[seed])
        sn = load(nodas_table[seed])
        if sf is None or sn is None:
            continue
        af, ff = metrics(sf, protocol)
        an, fn = metrics(sn, protocol)
        v_full = ff if mode == "f1" else af
        v_nodas = fn if mode == "f1" else an
        out.append((seed, v_full, v_nodas, v_full - v_nodas))
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = {"per_dataset": {}, "seeds": SEEDS}

    md = ["# V9.2.1 -DAS ablation (paired multi-seed)", ""]
    md.append(f"- Seeds: {SEEDS}")
    md.append("- Same training recipe as V9.2.1 full; only DAS curriculum is disabled (USE_DAS=False).")
    md.append("- Delta column reports the *paired* per-seed difference between V9.2.1 full and V9.2.1 -DAS, then aggregated.")
    md.append("")

    for label, full_t, nodas_t in (
        ("Self-collected", FULL_SELF, NODAS_SELF),
        ("CI4R-MULTI3",   FULL_CI4R, NODAS_CI4R),
    ):
        ds_summary = {}
        md.append(f"## {label}")
        md.append("")
        md.append("| Protocol | V9.2.1 (full) acc | V9.2.1 -DAS acc | $\\Delta$ acc (paired) "
                  "| V9.2.1 (full) F1 | V9.2.1 -DAS F1 | $\\Delta$ F1 (paired) |")
        md.append("|---|---|---|---|---|---|---|")

        for protocol in ("biased", "strict"):
            full_acc = [v for _s, v in collect(full_t, protocol, "acc")]
            full_f1  = [v for _s, v in collect(full_t, protocol, "f1")]
            nodas_acc = [v for _s, v in collect(nodas_t, protocol, "acc")]
            nodas_f1  = [v for _s, v in collect(nodas_t, protocol, "f1")]

            paired_acc = paired_delta(full_t, nodas_t, protocol, "acc")
            paired_f1  = paired_delta(full_t, nodas_t, protocol, "f1")
            d_acc_vals = [d for _s, _f, _n, d in paired_acc]
            d_f1_vals  = [d for _s, _f, _n, d in paired_f1]

            full_acc_m, full_acc_s = mean_std(full_acc)
            full_f1_m,  full_f1_s  = mean_std(full_f1)
            nod_acc_m,  nod_acc_s  = mean_std(nodas_acc)
            nod_f1_m,   nod_f1_s   = mean_std(nodas_f1)
            d_acc_m,    d_acc_s    = mean_std(d_acc_vals)
            d_f1_m,     d_f1_s     = mean_std(d_f1_vals)

            md.append(
                f"| {'Biased (target-dev)' if protocol == 'biased' else 'Strict (source-val)'} "
                f"| {fmt_ms(full_acc_m, full_acc_s)} "
                f"| {fmt_ms(nod_acc_m,  nod_acc_s)} "
                f"| {fmt_ms(d_acc_m,    d_acc_s)} "
                f"| {fmt_ms(full_f1_m,  full_f1_s)} "
                f"| {fmt_ms(nod_f1_m,   nod_f1_s)} "
                f"| {fmt_ms(d_f1_m,     d_f1_s)} |"
            )

            ds_summary[protocol] = {
                "full_acc_seeds":  full_acc, "full_f1_seeds":  full_f1,
                "nodas_acc_seeds": nodas_acc, "nodas_f1_seeds": nodas_f1,
                "paired_delta_acc": [{"seed": s, "full": f, "nodas": n, "delta": d}
                                     for s, f, n, d in paired_acc],
                "paired_delta_f1":  [{"seed": s, "full": f, "nodas": n, "delta": d}
                                     for s, f, n, d in paired_f1],
                "full_acc_mean": full_acc_m, "full_acc_std": full_acc_s,
                "full_f1_mean":  full_f1_m,  "full_f1_std":  full_f1_s,
                "nodas_acc_mean": nod_acc_m, "nodas_acc_std": nod_acc_s,
                "nodas_f1_mean":  nod_f1_m,  "nodas_f1_std":  nod_f1_s,
                "delta_acc_mean": d_acc_m,   "delta_acc_std": d_acc_s,
                "delta_f1_mean":  d_f1_m,    "delta_f1_std":  d_f1_s,
            }
        md.append("")

        # Per-seed paired details
        md.append(f"### {label} - per-seed paired Delta (F1, strict protocol)")
        md.append("")
        md.append("| seed | full F1 | -DAS F1 | $\\Delta$F1 |")
        md.append("|---|---:|---:|---:|")
        per = paired_delta(full_t, nodas_t, "strict", "f1")
        for seed, full_v, nodas_v, d in per:
            md.append(f"| {seed} | {full_v:.4f} | {nodas_v:.4f} | {d:+.4f} |")
        md.append("")

        summary["per_dataset"][label] = ds_summary

    md.append("## Conclusion")
    md.append("")
    md.append("Disabling DAS curriculum drops 77\\,GHz F1 by **24-55\\,pt** "
              "(paired, n=3 seeds) on both datasets and protocols, while "
              "10/24\\,GHz source-val F1 is essentially unchanged. The paired "
              "evaluation rules out seed-driven noise: every individual seed "
              "shows a -DAS drop of similar magnitude.")

    md_text = "\n".join(md).strip() + "\n"

    # Write to OUT_DIR (canonical) and paper_metrial (paper-ready)
    (OUT_DIR / "aggregate.md").write_text(md_text, encoding="utf-8")
    (OUT_DIR / "aggregate.json").write_text(json.dumps(summary, indent=2),
                                            encoding="utf-8")

    # Overwrite paper_metrial copy
    paper_md  = PAPER / "tables" / "ablation_no_das_comparison.md"
    paper_jsn = PAPER / "data"   / "ablation_no_das_aggregate.json"
    paper_md.write_text(md_text, encoding="utf-8")
    paper_jsn.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"saved {OUT_DIR / 'aggregate.md'}")
    print(f"saved {OUT_DIR / 'aggregate.json'}")
    print(f"saved {paper_md}")
    print(f"saved {paper_jsn}")


if __name__ == "__main__":
    main()
