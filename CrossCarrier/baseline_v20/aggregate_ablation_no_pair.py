"""Aggregate the V9.2.1 -pair ablation across 3 seeds, paired with
V9.2.1 full (DAS + pair) and V9.2.1 -DAS (no DAS, pair kept).

Produces the "loss-component ladder":
    floor (-DAS, pair)  vs  -pair (DAS only)  vs  full (DAS + pair)

Outputs:
    output/V9.2.1-Ablation-NoPair/aggregate.json
    output/V9.2.1-Ablation-NoPair/aggregate.md
    paper_metrial/M/ablation_no_pair.json
    paper_metrial/M/ablation_no_pair.md
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path("G:/zhanghe/output")
PAPER_M = Path("G:/zhanghe/paper_metrial/M")
OUT_DIR = ROOT / "V9.2.1-Ablation-NoPair"

SEEDS = [42, 1234, 7890]

# V9.2.1 full (DAS on, pair on)
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

# -DAS (DAS off, pair on)
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

# -pair  (DAS on, pair off)
NOPAIR_SELF = {
    42:   ROOT / "V9.2.1-KnownPeople-UnknownFreq" / "run_no_pair_seed42_strict",
    1234: ROOT / "V9.2.1-KnownPeople-UnknownFreq" / "run_no_pair_seed1234_strict",
    7890: ROOT / "V9.2.1-KnownPeople-UnknownFreq" / "run_no_pair_seed7890_strict",
}
NOPAIR_CI4R = {
    42:   ROOT / "V9.2.1-KnownPeople-UnknownFreq-CI4R" / "run_no_pair_seed42",
    1234: ROOT / "V9.2.1-KnownPeople-UnknownFreq-CI4R" / "run_no_pair_seed1234",
    7890: ROOT / "V9.2.1-KnownPeople-UnknownFreq-CI4R" / "run_no_pair_seed7890",
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


def build_panel(label, full_t, nodas_t, nopair_t):
    panel = {"label": label, "rows": {}}
    for protocol in ("biased", "strict"):
        full = collect(full_t, protocol)
        nodas = collect(nodas_t, protocol)
        nopair = collect(nopair_t, protocol)

        full_f1 = [r["f1"] for r in full]
        nodas_f1 = [r["f1"] for r in nodas]
        nopair_f1 = [r["f1"] for r in nopair]
        full_acc = [r["acc"] for r in full]
        nodas_acc = [r["acc"] for r in nodas]
        nopair_acc = [r["acc"] for r in nopair]

        panel["rows"][protocol] = {
            "full":   {"acc_mean_std": mean_std(full_acc),   "f1_mean_std": mean_std(full_f1),   "per_seed": full},
            "nodas":  {"acc_mean_std": mean_std(nodas_acc),  "f1_mean_std": mean_std(nodas_f1),  "per_seed": nodas},
            "nopair": {"acc_mean_std": mean_std(nopair_acc), "f1_mean_std": mean_std(nopair_f1), "per_seed": nopair},
        }
    return panel


def render_md(panels):
    lines = []
    lines.append("# Loss-component ladder: -pair vs -DAS vs full (V9.2.1 ablation)")
    lines.append("")
    lines.append("- Seeds: [42, 1234, 7890]")
    lines.append("- All three configurations share the V9.2.1 backbone, training recipe, "
                 "selection protocol, and seed set; they differ only in which loss "
                 "component is removed:")
    lines.append("  - **-DAS** (pair kept): DAS curriculum disabled, pair-consistency kept.")
    lines.append("  - **-pair** (DAS kept): pair-consistency loss disabled (pair_scale forced to 0); "
                 "DAS curriculum kept on.")
    lines.append("  - **full** (V9.2.1): both DAS curriculum and pair-consistency enabled.")
    lines.append("- Selection protocols: `biased` = argmax dev77 F1; `strict` = argmax source-val F1.")
    lines.append("")
    for panel in panels:
        lines.append(f"## {panel['label']}")
        lines.append("")
        lines.append("| Loss-component setting | Protocol | acc | F1 |")
        lines.append("|---|---|---|---|")
        for protocol in ("biased", "strict"):
            r = panel["rows"][protocol]
            lab_p = "Biased (target-dev)" if protocol == "biased" else "Strict (source-val)"
            for which, name in (
                ("nodas",  "-DAS (pair kept)"),
                ("nopair", "-pair (DAS kept)"),
                ("full",   "full (V9.2.1)"),
            ):
                acc_m, acc_s = r[which]["acc_mean_std"]
                f1_m,  f1_s  = r[which]["f1_mean_std"]
                lines.append(
                    f"| {name} | {lab_p} | {fmt_ms(acc_m, acc_s)} | {fmt_ms(f1_m, f1_s)} |"
                )
        lines.append("")
        lines.append(f"### {panel['label']} - per-seed F1, strict protocol")
        lines.append("")
        lines.append("| seed | -DAS | -pair | full |")
        lines.append("|---|---:|---:|---:|")
        for seed in SEEDS:
            r = panel["rows"]["strict"]
            def find(seed, lst):
                for x in lst:
                    if x["seed"] == seed:
                        return f"{x['f1']:.4f}"
                return "--"
            lines.append(
                f"| {seed} | {find(seed, r['nodas']['per_seed'])} "
                f"| {find(seed, r['nopair']['per_seed'])} "
                f"| {find(seed, r['full']['per_seed'])} |"
            )
        lines.append("")
    lines.append("## Reading")
    lines.append("")
    lines.append("Comparing -pair vs full quantifies how much the 10/24 GHz pair-consistency "
                 "loss adds beyond the DAS curriculum alone. A small gap supports treating pair "
                 "as an implementation detail; a large gap supports keeping pair as a co-equal "
                 "contribution alongside DAS in the framework description.")
    return "\n".join(lines).strip() + "\n"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_M.mkdir(parents=True, exist_ok=True)
    panels = [
        build_panel("Self-collected", FULL_SELF, NODAS_SELF, NOPAIR_SELF),
        build_panel("CI4R-MULTI3",   FULL_CI4R, NODAS_CI4R, NOPAIR_CI4R),
    ]

    md = render_md(panels)
    js = json.dumps(panels, indent=2, default=lambda o: list(o) if isinstance(o, tuple) else o)

    (OUT_DIR / "aggregate.md").write_text(md, encoding="utf-8")
    (OUT_DIR / "aggregate.json").write_text(js, encoding="utf-8")

    paper_md = PAPER_M / "ablation_no_pair.md"
    paper_jsn = PAPER_M / "ablation_no_pair.json"
    paper_md.write_text(md, encoding="utf-8")
    paper_jsn.write_text(js, encoding="utf-8")

    print(f"saved {OUT_DIR / 'aggregate.md'}")
    print(f"saved {paper_md}")


if __name__ == "__main__":
    main()
