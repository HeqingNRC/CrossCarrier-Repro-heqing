"""Post-hoc strict-protocol analysis for a V9.2.1 run.

Reads `<run-dir>/reports/history.json` (which now contains test77_f1_ema and
test77_acc_ema per epoch) and computes the test number under TWO checkpoint
selection protocols:

  1. Biased (V9.2.1 default):
       best_ep = argmax_e dev77_f1_ema[e]    (uses 77GHz target labels)
       report  = test77 at best_ep
  2. Unbiased (strict source-val):
       best_ep = argmax_e val_f1_ema[e]      (uses ONLY 10/24 GHz val labels)
       report  = test77 at best_ep

The `test77_f1_ema[e]` numbers are pure forward-only evaluations: they were
recorded during training but did NOT influence gradients or selection. So
the unbiased protocol's reported number contains zero target-domain leakage.

Usage
-----
    python baseline_v9_2_1/strict_analysis.py \
        --run-dir G:/zhanghe/output/V9.2.1-KnownPeople-UnknownFreq/run_002_strict

Outputs
-------
    <run-dir>/reports/strict_analysis.json
    <run-dir>/reports/strict_analysis.md
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True,
                    help="V9.2.1 run-dir (must contain reports/history.json)")
    return ap.parse_args()


def best_epoch(history, key):
    """Return (epoch_index, max_value) where epoch_index is 1-based."""
    vals = [(i, h.get(key, 0.0) or 0.0) for i, h in enumerate(history)]
    if not vals:
        return None, 0.0
    i, v = max(vals, key=lambda x: x[1])
    return history[i].get("epoch", i + 1), v


def lookup_at_epoch(history, target_epoch, *keys):
    for h in history:
        if int(h.get("epoch", -1)) == int(target_epoch):
            return [h.get(k) for k in keys]
    return [None] * len(keys)


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    hist_path = run_dir / "reports" / "history.json"
    if not hist_path.exists():
        raise SystemExit(f"history.json not found at {hist_path}; "
                         f"is this a completed V9.2.1 run-dir?")

    with open(hist_path, encoding="utf-8") as f:
        history = json.load(f)
    if not history:
        raise SystemExit("history.json is empty")

    print(f"Loaded {len(history)} epochs from {hist_path}")
    print(f"Available per-epoch fields: {sorted(history[0].keys())}")
    print()

    if "test77_f1_ema" not in history[0]:
        raise SystemExit(
            "history.json does NOT contain 'test77_f1_ema' per epoch. "
            "This run was trained before the per-epoch test logging was added. "
            "You must rerun training with the updated baseline_v9_2_1/train.py "
            "(under run-dir 'run_002_strict' or similar)."
        )

    # ----- 1. Biased: dev77_f1_ema selection -----
    bias_ep, bias_dev = best_epoch(history, "dev77_f1_ema")
    bias_test_acc, bias_test_f1, bias_val_f1, bias_dev_f1 = lookup_at_epoch(
        history, bias_ep,
        "test77_acc_ema", "test77_f1_ema", "val_f1_ema", "dev77_f1_ema",
    )

    # ----- 2. Unbiased: val_f1_ema selection -----
    strict_ep, strict_val = best_epoch(history, "val_f1_ema")
    strict_test_acc, strict_test_f1, strict_val_f1, strict_dev_f1 = lookup_at_epoch(
        history, strict_ep,
        "test77_acc_ema", "test77_f1_ema", "val_f1_ema", "dev77_f1_ema",
    )

    # ----- (Optional comparison) Last epoch -----
    last = history[-1]
    last_ep = int(last.get("epoch", len(history)))
    last_test_acc = last.get("test77_acc_ema")
    last_test_f1 = last.get("test77_f1_ema")

    summary = {
        "run_dir": str(run_dir),
        "n_epochs": len(history),
        "biased_dev77_selection": {
            "selection_metric": "dev77_f1_ema",
            "best_epoch": int(bias_ep),
            "selected_dev77_f1_ema": float(bias_dev_f1) if bias_dev_f1 is not None else None,
            "reported_val_f1_ema": float(bias_val_f1) if bias_val_f1 is not None else None,
            "test77_acc": float(bias_test_acc) if bias_test_acc is not None else None,
            "test77_f1":  float(bias_test_f1)  if bias_test_f1  is not None else None,
        },
        "strict_sourceval_selection": {
            "selection_metric": "val_f1_ema",
            "best_epoch": int(strict_ep),
            "selected_val_f1_ema": float(strict_val_f1) if strict_val_f1 is not None else None,
            "reported_dev77_f1_ema": float(strict_dev_f1) if strict_dev_f1 is not None else None,
            "test77_acc": float(strict_test_acc) if strict_test_acc is not None else None,
            "test77_f1":  float(strict_test_f1)  if strict_test_f1  is not None else None,
        },
        "last_epoch_no_selection": {
            "epoch": last_ep,
            "test77_acc": float(last_test_acc) if last_test_acc is not None else None,
            "test77_f1":  float(last_test_f1)  if last_test_f1  is not None else None,
        },
        "bias_estimate_test77_f1": (
            (float(bias_test_f1) - float(strict_test_f1))
            if (bias_test_f1 is not None and strict_test_f1 is not None)
            else None
        ),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    out_json = run_dir / "reports" / "strict_analysis.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved {out_json}")

    # ----- Markdown report -----
    md = []
    md.append("# Strict-protocol vs target-dev-protocol comparison")
    md.append("")
    md.append(f"- Generated: {summary['generated_at']}")
    md.append(f"- Run dir: `{summary['run_dir']}`")
    md.append(f"- Epochs analysed: {summary['n_epochs']}")
    md.append("")
    md.append("## Selection protocols")
    md.append("")
    md.append("| Protocol | Selection metric | Best epoch | dev77 F1 (EMA) | val F1 (EMA) | **test77 acc** | **test77 macro-F1** |")
    md.append("|---|---|---:|---:|---:|---:|---:|")
    b = summary["biased_dev77_selection"]
    s = summary["strict_sourceval_selection"]
    le = summary["last_epoch_no_selection"]

    def fmt(v): return f"{v:.4f}" if isinstance(v, (int, float)) and v is not None else "--"

    md.append(
        f"| **Biased** (target-dev) | dev77_f1_ema | {b['best_epoch']} "
        f"| {fmt(b['selected_dev77_f1_ema'])} | {fmt(b['reported_val_f1_ema'])} "
        f"| **{fmt(b['test77_acc'])}** | **{fmt(b['test77_f1'])}** |"
    )
    md.append(
        f"| **Strict** (source-val) | val_f1_ema | {s['best_epoch']} "
        f"| {fmt(s['reported_dev77_f1_ema'])} | {fmt(s['selected_val_f1_ema'])} "
        f"| **{fmt(s['test77_acc'])}** | **{fmt(s['test77_f1'])}** |"
    )
    md.append(
        f"| Last epoch (no selection) | -- | {le['epoch']} | -- | -- "
        f"| {fmt(le['test77_acc'])} | {fmt(le['test77_f1'])} |"
    )
    md.append("")

    if summary["bias_estimate_test77_f1"] is not None:
        bias = summary["bias_estimate_test77_f1"]
        md.append(f"**Bias estimate** (biased - strict, on test77 macro-F1): "
                  f"`{bias:+.4f}`")
        md.append("")
        if bias > 0:
            md.append("Interpretation: the biased (dev77-selection) protocol over-reports "
                      f"by approximately {bias:+.4f} F1 points compared to a strict "
                      "source-val selection that uses zero target-domain labels.")
        else:
            md.append("Interpretation: in this run, target-dev selection happens to "
                      "agree with strict source-val selection (or undershoot it). The "
                      "implied bias is small.")
        md.append("")

    md.append("## How to read this")
    md.append("")
    md.append("- The training loop evaluates 77GHz dev (140 imgs), 77GHz final test "
              "(278 imgs) and 10/24 GHz source val (162 imgs) every epoch under EMA. "
              "All three are pure forward passes; gradients never see them.")
    md.append("- The **only** path by which a target signal influences the reported "
              "number is **the choice of which epoch's checkpoint to evaluate**.")
    md.append("- The strict protocol picks that epoch using `val_f1_ema` "
              "(source-val only -> no 77GHz labels in selection).")
    md.append("- The biased protocol picks that epoch using `dev77_f1_ema` "
              "(uses 140 labelled 77GHz dev images).")
    md.append("- Difference between the two test77 numbers = *upper bound* on how "
              "much the biased protocol's number was inflated by target-dev selection.")
    md.append("")

    out_md = run_dir / "reports" / "strict_analysis.md"
    out_md.write_text("\n".join(md).strip() + "\n", encoding="utf-8")
    print(f"saved {out_md}")

    # ----- Pretty print to stdout for quick read -----
    print()
    print("=" * 70)
    print(f" Biased  (dev77 selection) : ep={b['best_epoch']:3d}  "
          f"test77 acc={fmt(b['test77_acc'])}  F1={fmt(b['test77_f1'])}")
    print(f" Strict  (val   selection) : ep={s['best_epoch']:3d}  "
          f"test77 acc={fmt(s['test77_acc'])}  F1={fmt(s['test77_f1'])}")
    print(f" Last    (no   selection)  : ep={le['epoch']:3d}  "
          f"test77 acc={fmt(le['test77_acc'])}  F1={fmt(le['test77_f1'])}")
    if summary["bias_estimate_test77_f1"] is not None:
        print(f" Bias estimate (biased - strict on F1): "
              f"{summary['bias_estimate_test77_f1']:+.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
