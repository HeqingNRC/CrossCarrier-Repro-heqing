"""Export image-based evaluation figures for a completed V9.2.1 run.

Outputs:
  - confusion matrix image for 77GHz final test
  - report-table image with accuracy / precision / recall / F1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

import config
from v9_2_1lib import (
    ModelEMA,
    TimmBackboneV921,
    evaluate,
    load_or_build_test77_dev_final,
    make_eval_loader,
)


def parse_args():
    ap = argparse.ArgumentParser(description="Export V9.2.1 result figures")
    ap.add_argument("--run-dir", required=True, help="completed run directory")
    ap.add_argument("--ckpt", choices=["ema", "best", "last"], default="ema")
    return ap.parse_args()


def amp_dtype():
    return torch.bfloat16 if config.AMP_DTYPE == "bf16" else torch.float16


def safe_name(backbone):
    return backbone.replace("/", "_").replace(".", "_").replace(":", "_")


def resolve_ckpt_path(run_dir: Path, backbone: str, ckpt: str) -> tuple[Path, str]:
    if ckpt == "ema":
        return run_dir / "checkpoints" / f"{safe_name(backbone)}_best_ema.pt", "ema"
    if ckpt == "best":
        return run_dir / "checkpoints" / f"{safe_name(backbone)}_best_live.pt", "live"
    return run_dir / "checkpoints" / f"{safe_name(backbone)}_last.pt", "last"


def load_model_from_ckpt(run_dir: Path, ckpt_choice: str, device: torch.device):
    reports_dir = run_dir / "reports"
    eval_json = reports_dir / "eval_ema.json"
    backbone = config.DEFAULT_BACKBONE
    if eval_json.exists():
        with open(eval_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        backbone = payload.get("checkpoint", backbone)
    # We only need the safe filename stem for lookup; real backbone comes from ckpt metadata.
    backbone_guess = config.DEFAULT_BACKBONE
    ckpt_path, selector = resolve_ckpt_path(run_dir, backbone_guess, ckpt_choice)
    if not ckpt_path.exists() and ckpt_choice == "ema":
        fallback = run_dir / "checkpoints" / f"{safe_name(backbone_guess)}_best_live.pt"
        if fallback.exists():
            ckpt_path = fallback
            selector = "live"
        else:
            ckpt_path = run_dir / "checkpoints" / f"{safe_name(backbone_guess)}_last.pt"
            selector = "last"
    elif not ckpt_path.exists() and ckpt_choice == "best":
        ckpt_path = run_dir / "checkpoints" / f"{safe_name(backbone_guess)}_last.pt"
        selector = "last"

    ckpt = torch.load(ckpt_path, map_location=device)
    backbone_name = ckpt.get("backbone", config.DEFAULT_BACKBONE)
    adapter_mode = ckpt.get("adapter_mode", config.BACKBONE_TUNE_MODE)

    model = TimmBackboneV921(
        backbone_name,
        config.NUM_CLASSES,
        adapter_mode=adapter_mode,
    ).to(device)
    ema = ModelEMA(model, config.EMA_DECAY)

    if selector == "ema":
        ema.load_state_dict(ckpt["ema"])
        ema.copy_to(model)
    elif selector == "live":
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt["model"])
        if "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])

    model.eval()
    return model, {
        "ckpt_path": ckpt_path,
        "selector": selector,
        "backbone": backbone_name,
        "adapter_mode": adapter_mode,
        "epoch": ckpt.get("epoch"),
    }


def build_metric_tables(labels: np.ndarray, preds: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    labels_idx = list(range(config.NUM_CLASSES))
    class_names = list(config.CLASSES)

    prec, rec, f1, support = precision_recall_fscore_support(
        labels, preds, labels=labels_idx, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    acc = accuracy_score(labels, preds)

    per_class_df = pd.DataFrame(
        {
            "Class": class_names,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "Support": support,
        }
    )

    summary_df = pd.DataFrame(
        [
            {
                "Accuracy": acc,
                "Macro Precision": macro_p,
                "Macro Recall": macro_r,
                "Macro F1": macro_f1,
                "Weighted Precision": weighted_p,
                "Weighted Recall": weighted_r,
                "Weighted F1": weighted_f1,
                "Samples": int(len(labels)),
            }
        ]
    )

    summary = {
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "samples": int(len(labels)),
    }
    return summary_df, per_class_df, summary


def _format_table(df: pd.DataFrame, float_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in float_cols:
        if col in out.columns:
            out[col] = out[col].map(lambda x: f"{float(x):.4f}")
    return out


def render_report_table(out_path: Path, summary_df: pd.DataFrame, per_class_df: pd.DataFrame, meta: dict) -> None:
    summary_disp = _format_table(
        summary_df,
        [
            "Accuracy",
            "Macro Precision",
            "Macro Recall",
            "Macro F1",
            "Weighted Precision",
            "Weighted Recall",
            "Weighted F1",
        ],
    )
    per_class_disp = _format_table(per_class_df, ["Precision", "Recall", "F1"])
    per_class_disp["Support"] = per_class_disp["Support"].astype(int).astype(str)

    fig = plt.figure(figsize=(16, 8), dpi=220)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 2.4])
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax0.axis("off")
    ax1.axis("off")

    title = (
        "V9.2.1 77GHz Final-Test Report Table\n"
        f"checkpoint={meta['ckpt_path'].name} ({meta['selector']}), "
        f"adapter={meta['adapter_mode']}, epoch={meta['epoch']}"
    )
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

    ax0.text(0.0, 1.12, "Overall Metrics", fontsize=13, fontweight="bold", transform=ax0.transAxes)
    tbl0 = ax0.table(
        cellText=summary_disp.values,
        colLabels=list(summary_disp.columns),
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    tbl0.auto_set_font_size(False)
    tbl0.set_fontsize(10)
    tbl0.scale(1.1, 1.8)
    for (r, c), cell in tbl0.get_celld().items():
        cell.set_edgecolor("#B8C4D6")
        if r == 0:
            cell.set_facecolor("#DCEAF7")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#F8FBFE")

    ax1.text(0.0, 1.05, "Per-Class Metrics", fontsize=13, fontweight="bold", transform=ax1.transAxes)
    tbl1 = ax1.table(
        cellText=per_class_disp.values,
        colLabels=list(per_class_disp.columns),
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    tbl1.auto_set_font_size(False)
    tbl1.set_fontsize(10)
    tbl1.scale(1.0, 1.65)
    for (r, c), cell in tbl1.get_celld().items():
        cell.set_edgecolor("#B8C4D6")
        if r == 0:
            cell.set_facecolor("#DCEAF7")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#FFFFFF" if r % 2 == 1 else "#F8FBFE")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def render_confusion_matrix(out_path: Path, labels: np.ndarray, preds: np.ndarray, meta: dict, summary: dict) -> None:
    cm = confusion_matrix(labels, preds, labels=list(range(config.NUM_CLASSES)))
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(10.5, 8.5), dpi=220)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized recall", rotation=90)

    ax.set_xticks(np.arange(config.NUM_CLASSES))
    ax.set_yticks(np.arange(config.NUM_CLASSES))
    ax.set_xticklabels(config.CLASSES, rotation=35, ha="right")
    ax.set_yticklabels(config.CLASSES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(
        "V9.2.1 77GHz Final-Test Confusion Matrix\n"
        f"acc={summary['accuracy']:.4f}, macro-F1={summary['macro_f1']:.4f}, "
        f"ckpt={meta['ckpt_path'].name}",
        fontsize=14,
        pad=14,
    )

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm_norm[i, j] >= 0.5 else "#1F2937"
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n{cm_norm[i, j]:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=9,
                fontweight="bold" if i == j else "normal",
            )

    ax.set_xlim(-0.5, config.NUM_CLASSES - 0.5)
    ax.set_ylim(config.NUM_CLASSES - 0.5, -0.5)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_model_from_ckpt(run_dir, args.ckpt, device)

    _dev_df, test77_final_df, _split_info = load_or_build_test77_dev_final(run_dir=run_dir)
    test_loader = make_eval_loader(test77_final_df)
    result = evaluate(model, test_loader, device, amp_dtype())
    preds = np.asarray(result["preds"])
    labels = np.asarray(result["labels"])

    summary_df, per_class_df, summary = build_metric_tables(labels, preds)

    stem = f"test77_final_{meta['selector']}"
    cm_path = reports_dir / f"{stem}_confusion_matrix.png"
    table_path = reports_dir / f"{stem}_report_table.png"
    json_path = reports_dir / f"{stem}_report_metrics.json"

    render_confusion_matrix(cm_path, labels, preds, meta, summary)
    render_report_table(table_path, summary_df, per_class_df, meta)

    payload = {
        "meta": {
            "checkpoint": meta["ckpt_path"].name,
            "selector": meta["selector"],
            "backbone": meta["backbone"],
            "adapter_mode": meta["adapter_mode"],
            "epoch": meta["epoch"],
        },
        "overall": summary,
        "per_class": per_class_df.to_dict(orient="records"),
        "paths": {
            "confusion_matrix": str(cm_path),
            "report_table": str(table_path),
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"saved confusion matrix: {cm_path}")
    print(f"saved report table: {table_path}")
    print(f"saved metrics json: {json_path}")


if __name__ == "__main__":
    main()
