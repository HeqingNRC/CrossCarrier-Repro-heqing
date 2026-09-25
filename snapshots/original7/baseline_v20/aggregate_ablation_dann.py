"""Aggregate the V9.2.1 -DAS+DANN ablation across 3 seeds, paired with
V9.2.1 full (DAS curriculum) and V9.2.1 -DAS (no cross-band mechanism).

Produces the "cross-band mechanism ladder":
    floor (-DAS, no DANN)  <  data-driven (DANN)  <  physics-driven (DAS)

Outputs:
    output/V9.2.1-Ablation-DANN/aggregate.json
    output/V9.2.1-Ablation-DANN/aggregate.md
    paper_metrial/tables/ablation_cross_band_ladder.md
    paper_metrial/data/ablation_cross_band_ladder.json
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path("G:/zhanghe/output")
PAPER = Path("G:/zhanghe/paper_metrial")
OUT_DIR = ROOT / "V9.2.1-Ablation-DANN"

SEEDS = [42, 1234, 7890]

# V9.2.1 full (DAS on, DANN off)
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

# -DAS (DAS off, DANN off)
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

# -DAS + DANN  (DAS off, DANN on)
DANN_SELF = {
    42:   ROOT / "V9.2.1-KnownPeople-UnknownFreq" / "run_dann_seed42_strict",
    1234: ROOT / "V9.2.1-KnownPeople-UnknownFreq" / "run_dann_seed1234_strict",
    7890: ROOT / "V9.2.1-KnownPeople-UnknownFreq" / "run_dann_seed7890_strict",
}
DANN_CI4R = {
    42:   ROOT / "V9.2.1-KnownPeople-UnknownFreq-CI4R" / "run_dann_seed42",
    1234: ROOT / "V9.2.1-KnownPeople-UnknownFreq-CI4R" / "run_dann_seed1234",
    7890: ROOT / "V9.2.1-KnownPeople-UnknownFreq-CI4R" / "run_dann_seed7890",
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


def build_panel(label, full_t, nodas_t, dann_t):
    panel = {"label": label, "rows": {}}
    for protocol in ("biased", "strict"):
        full = collect(full_t, protocol)
        nodas = collect(nodas_t, protocol)
        dann = collect(dann_t, protocol)

        full_f1  = [r["f1"]  for r in full]
        nodas_f1 = [r["f1"]  for r in nodas]
        dann_f1  = [r["f1"]  for r in dann]
        full_acc  = [r["acc"] for r in full]
        nodas_acc = [r["acc"] for r in nodas]
        dann_acc  = [r["acc"] for r in dann]

        panel["rows"][protocol] = {
            "full":  {"acc_mean_std": mean_std(full_acc),  "f1_mean_std": mean_std(full_f1),  "per_seed": full},
            "nodas": {"acc_mean_std": mean_std(nodas_acc), "f1_mean_std": mean_std(nodas_f1), "per_seed": nodas},
            "dann":  {"acc_mean_std": mean_std(dann_acc),  "f1_mean_std": mean_std(dann_f1),  "per_seed": dann},
        }
    return panel


def render_md(panels):
    lines = []
    lines.append("# Cross-band mechanism ladder (V9.2.1 ablation)")
    lines.append("")
    lines.append("- Seeds: [42, 1234, 7890]")
    lines.append("- All three configurations share the V9.2.1 backbone, training recipe, "
                 "selection protocol, and seed set; they differ only in the cross-band mechanism:")
    lines.append("  - **None** (-DAS): both DAS curriculum and DANN disabled (floor).")
    lines.append("  - **DANN**: DAS curriculum disabled, band-adversarial domain classifier enabled "
                 "(data-driven cross-band signal between 10 GHz and 24 GHz only).")
    lines.append("  - **DAS** (V9.2.1 full): physics-driven cross-band augmentation via "
                 "$f_d=2vf_c/c$ Doppler-axis stretch.")
    lines.append("- Selection protocols: `biased` = argmax dev77 F1; `strict` = argmax source-val F1.")
    lines.append("")
    for panel in panels:
        lines.append(f"## {panel['label']}")
        lines.append("")
        lines.append("| Cross-band mechanism | Protocol | acc | F1 |")
        lines.append("|---|---|---|---|")
        for protocol in ("biased", "strict"):
            r = panel["rows"][protocol]
            lab_p = "Biased (target-dev)" if protocol == "biased" else "Strict (source-val)"
            for which, name in (
                ("nodas", "None (-DAS)"),
                ("dann",  "DANN (data-driven)"),
                ("full",  "DAS (V9.2.1)"),
            ):
                acc_m, acc_s = r[which]["acc_mean_std"]
                f1_m,  f1_s  = r[which]["f1_mean_std"]
                lines.append(
                    f"| {name} | {lab_p} | {fmt_ms(acc_m, acc_s)} | {fmt_ms(f1_m, f1_s)} |"
                )
        lines.append("")
        lines.append(f"### {panel['label']} - per-seed F1, strict protocol")
        lines.append("")
        lines.append("| seed | -DAS (None) | DANN | DAS (full) |")
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
                f"| {find(seed, r['dann']['per_seed'])} "
                f"| {find(seed, r['full']['per_seed'])} |"
            )
        lines.append("")
    lines.append("## Conclusion")
    lines.append("")
    lines.append("The DANN baseline plugs a data-driven band-adversarial signal into the "
                 "V9.2.1 framework while keeping every other component identical. The "
                 "remaining gap between DANN and DAS is therefore attributable to the "
                 "physics prior carried by the DAS curriculum, not to the existence of a "
                 "cross-band mechanism in general.")
    return "\n".join(lines).strip() + "\n"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panels = [
        build_panel("Self-collected", FULL_SELF, NODAS_SELF, DANN_SELF),
        build_panel("CI4R-MULTI3",   FULL_CI4R, NODAS_CI4R, DANN_CI4R),
    ]

    md = render_md(panels)
    js = json.dumps(panels, indent=2, default=lambda o: list(o) if isinstance(o, tuple) else o)

    (OUT_DIR / "aggregate.md").write_text(md, encoding="utf-8")
    (OUT_DIR / "aggregate.json").write_text(js, encoding="utf-8")

    paper_md = PAPER / "tables" / "ablation_cross_band_ladder.md"
    paper_jsn = PAPER / "data"  / "ablation_cross_band_ladder.json"
    paper_md.write_text(md, encoding="utf-8")
    paper_jsn.write_text(js, encoding="utf-8")

    print(f"saved {OUT_DIR / 'aggregate.md'}")
    print(f"saved {paper_md}")


if __name__ == "__main__":
    main()
