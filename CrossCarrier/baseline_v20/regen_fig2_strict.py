"""Re-render Fig 2 confusion matrices with source-val (strict) selected
checkpoint, and emit a JSON summary of the dominant confusion pairs so we
can verify the paper's Sec III-E claims (Bend->Pick, Kneel<->Sit) under
strict selection.

Inputs are the V9.2.1 full runs for seed=42:
  Self: output/V9.2.1-KnownPeople-UnknownFreq/run_002_strict (best_live.pt)
  CI4R: output/V9.2.1-KnownPeople-UnknownFreq-CI4R/run_001 (best_live.pt)

Outputs (all under paper_metrial/M/):
  fig2_strict_self_cm.png + .pdf      -- Self panel, no acc/F1 in title
  fig2_strict_self_metrics.json       -- per-class + dominant confusion pairs
  fig2_strict_ci4r_cm.png + .pdf      -- CI4R panel, no acc/F1 in title
  fig2_strict_ci4r_metrics.json
  fig2_strict_combined.pdf            -- 2-panel side-by-side for Fig 2
  section_iiie_pair_check.md          -- whether the paper claims still hold
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--self-run-dir",
        default="G:/zhanghe/output/V9.2.1-KnownPeople-UnknownFreq/run_002_strict",
    )
    ap.add_argument(
        "--ci4r-run-dir",
        default="G:/zhanghe/output/V9.2.1-KnownPeople-UnknownFreq-CI4R/run_001",
    )
    ap.add_argument(
        "--out-dir",
        default="G:/zhanghe/paper_metrial/M",
    )
    ap.add_argument("--top-k-pairs", type=int, default=6)
    return ap.parse_args()


def safe_name(backbone):
    return backbone.replace("/", "_").replace(".", "_").replace(":", "_")


def amp_dtype():
    import config
    return torch.bfloat16 if config.AMP_DTYPE == "bf16" else torch.float16


def load_best_live(run_dir: Path, device: torch.device):
    """Load the source-val (strict) selected checkpoint."""
    import config
    from v9_2_1lib import ModelEMA, TimmBackboneV921

    ckpt_path = (
        run_dir
        / "checkpoints"
        / f"{safe_name(config.DEFAULT_BACKBONE)}_best_live.pt"
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(f"missing strict checkpoint: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location=device)
    backbone_name = ckpt.get("backbone", config.DEFAULT_BACKBONE)
    adapter_mode = ckpt.get("adapter_mode", config.BACKBONE_TUNE_MODE)

    model = TimmBackboneV921(
        backbone_name,
        config.NUM_CLASSES,
        adapter_mode=adapter_mode,
    ).to(device)
    # best_live ckpt stores the live (non-EMA) weights under "model"
    model.load_state_dict(ckpt["model"])
    model.eval()

    return model, {
        "ckpt_path": ckpt_path,
        "selector": "live",
        "backbone": backbone_name,
        "adapter_mode": adapter_mode,
        "epoch": ckpt.get("epoch"),
    }


def run_inference(model, run_dir: Path, device: torch.device):
    from v9_2_1lib import (
        evaluate,
        load_or_build_test77_dev_final,
        make_eval_loader,
    )

    _dev_df, test_df, _split_info = load_or_build_test77_dev_final(run_dir=run_dir)
    loader = make_eval_loader(test_df)
    result = evaluate(model, loader, device, amp_dtype())
    return np.asarray(result["preds"]), np.asarray(result["labels"])


def compute_metrics(labels: np.ndarray, preds: np.ndarray):
    import config
    labels_idx = list(range(config.NUM_CLASSES))
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
    return {
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "samples": int(len(labels)),
        "per_class": [
            {
                "class": cls,
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "support": int(s),
            }
            for cls, p, r, f, s in zip(config.CLASSES, prec, rec, f1, support)
        ],
    }


def render_cm(out_pdf: Path, out_png: Path, labels: np.ndarray, preds: np.ndarray, panel_label: str):
    import config
    cm = confusion_matrix(labels, preds, labels=list(range(config.NUM_CLASSES)))
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(6.6, 6.0), dpi=240)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized recall", rotation=90)

    ax.set_xticks(np.arange(config.NUM_CLASSES))
    ax.set_yticks(np.arange(config.NUM_CLASSES))
    ax.set_xticklabels(config.CLASSES, rotation=35, ha="right")
    ax.set_yticklabels(config.CLASSES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    # Per agreement: NO acc/F1 numbers in title; only the panel label.
    # Source-only checkpoint selection is now stated in the figure caption.
    ax.set_title(panel_label, fontsize=13, pad=10)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm_norm[i, j] >= 0.5 else "#1F2937"
            ax.text(
                j, i,
                f"{cm[i, j]}\n{cm_norm[i, j]:.2f}",
                ha="center", va="center",
                color=color, fontsize=8,
                fontweight="bold" if i == j else "normal",
            )

    ax.set_xlim(-0.5, config.NUM_CLASSES - 0.5)
    ax.set_ylim(config.NUM_CLASSES - 0.5, -0.5)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return cm


def render_combined(out_pdf: Path, panels: list[tuple[str, np.ndarray]]):
    import config
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(6.6 * n, 6.0), dpi=240)
    if n == 1:
        axes = [axes]
    for ax, (label, cm) in zip(axes, panels):
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_xticks(np.arange(config.NUM_CLASSES))
        ax.set_yticks(np.arange(config.NUM_CLASSES))
        ax.set_xticklabels(config.CLASSES, rotation=35, ha="right")
        ax.set_yticklabels(config.CLASSES)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(label, fontsize=13, pad=10)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if cm_norm[i, j] >= 0.5 else "#1F2937"
                ax.text(
                    j, i,
                    f"{cm[i, j]}\n{cm_norm[i, j]:.2f}",
                    ha="center", va="center",
                    color=color, fontsize=8,
                    fontweight="bold" if i == j else "normal",
                )
        ax.set_xlim(-0.5, config.NUM_CLASSES - 0.5)
        ax.set_ylim(config.NUM_CLASSES - 0.5, -0.5)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def dominant_confusion_pairs(cm: np.ndarray, top_k: int):
    """Return top-K off-diagonal confusion pairs by row-normalized rate."""
    import config
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)
    pairs = []
    for i in range(config.NUM_CLASSES):
        for j in range(config.NUM_CLASSES):
            if i == j:
                continue
            if cm[i, j] == 0:
                continue
            pairs.append({
                "true_class": config.CLASSES[i],
                "pred_class": config.CLASSES[j],
                "count": int(cm[i, j]),
                "rate": float(cm_norm[i, j]),
            })
    pairs.sort(key=lambda x: x["rate"], reverse=True)
    return pairs[:top_k]


def evaluate_one(run_dir: Path, panel_label: str, slug: str, out_dir: Path, top_k: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_best_live(run_dir, device)
    preds, labels = run_inference(model, run_dir, device)
    metrics = compute_metrics(labels, preds)

    cm_pdf = out_dir / f"fig2_strict_{slug}_cm.pdf"
    cm_png = out_dir / f"fig2_strict_{slug}_cm.png"
    cm = render_cm(cm_pdf, cm_png, labels, preds, panel_label=panel_label)

    pairs = dominant_confusion_pairs(cm, top_k)

    payload = {
        "panel": panel_label,
        "run_dir": str(run_dir),
        "checkpoint": str(meta["ckpt_path"]),
        "selector": meta["selector"],
        "backbone": meta["backbone"],
        "adapter_mode": meta["adapter_mode"],
        "epoch": meta["epoch"],
        "metrics": metrics,
        "dominant_confusion_pairs": pairs,
    }
    json_path = out_dir / f"fig2_strict_{slug}_metrics.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Free GPU before next call
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload, cm


def write_pair_check(payloads: dict, out_md: Path):
    """Check the paper's two §III-E claims under strict selection."""
    claims = [
        ("Bend", "Pick", "Bend -> Pick"),
        ("Kneel", "Sit", "Kneel -> Sit"),
        ("Sit", "Kneel", "Sit -> Kneel"),
    ]

    lines = []
    lines.append("# §III-E confusion-pair check under strict selection")
    lines.append("")
    lines.append("Paper claim (current §III-E): the dominant confusions are **Bend->Pick** "
                 "and **Kneel<->Sit**. This file verifies whether those pairs still dominate "
                 "after switching from biased (target-dev) to strict (source-val) checkpoint "
                 "selection. If they don't, §III-E needs a one-line rewrite.")
    lines.append("")

    for slug, payload in payloads.items():
        lines.append(f"## {payload['panel']}")
        lines.append("")
        lines.append(f"Checkpoint: `{Path(payload['checkpoint']).name}` (epoch {payload['epoch']}, source-val selected)")
        lines.append(f"Macro-F1: {payload['metrics']['macro_f1']:.4f}, "
                     f"Accuracy: {payload['metrics']['accuracy']:.4f}, "
                     f"Samples: {payload['metrics']['samples']}")
        lines.append("")
        lines.append("**Top off-diagonal confusion pairs (by row-normalized rate):**")
        lines.append("")
        lines.append("| rank | true -> pred | count | rate |")
        lines.append("|---:|---|---:|---:|")
        for i, p in enumerate(payload["dominant_confusion_pairs"], 1):
            lines.append(f"| {i} | {p['true_class']} -> {p['pred_class']} | {p['count']} | {p['rate']:.3f} |")
        lines.append("")

        lines.append("**Paper claim verification:**")
        lines.append("")
        lines.append("| claim | observed rate | rank | holds? |")
        lines.append("|---|---:|---:|---|")
        pair_index = {(p["true_class"], p["pred_class"]): (i + 1, p["rate"])
                      for i, p in enumerate(payload["dominant_confusion_pairs"])}
        for true_cls, pred_cls, label in claims:
            if (true_cls, pred_cls) in pair_index:
                rank, rate = pair_index[(true_cls, pred_cls)]
                holds = "yes (top-3)" if rank <= 3 else ("borderline" if rank <= 5 else "no")
                lines.append(f"| {label} | {rate:.3f} | {rank} | {holds} |")
            else:
                lines.append(f"| {label} | -- | -- | not in top-K |")
        lines.append("")

    lines.append("## Recommendation")
    lines.append("")
    lines.append("If both \"Bend -> Pick\" and at least one direction of \"Kneel <-> Sit\" appear "
                 "in the top-3 (or have rate >= 0.20) on at least one of the two panels, the §III-E "
                 "wording can stay as-is. Otherwise, replace the offending claim with whichever "
                 "pair actually dominates per the tables above.")

    out_md.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Self uses the default known_people_unknown_freq task.
    os.environ.pop("V921_TASK_DIR_NAME", None)
    os.environ.pop("V921_RUNS_DIR_TAG", None)
    os.environ.pop("V921_EXPERIMENT_NAME", None)

    # Reload config under fresh env to get correct task dir.
    import importlib
    import config as _cfg
    importlib.reload(_cfg)

    self_payload, self_cm = evaluate_one(
        Path(args.self_run_dir),
        panel_label="(a) Self (10/24 GHz + NRC augmented)",
        slug="self",
        out_dir=out_dir,
        top_k=args.top_k_pairs,
    )

    # CI4R uses the ci4r task dir.
    os.environ["V921_TASK_DIR_NAME"] = "ci4r_known_people_unknown_freq"
    os.environ["V921_RUNS_DIR_TAG"] = "V9.2.1-KnownPeople-UnknownFreq-CI4R"
    os.environ["V921_EXPERIMENT_NAME"] = "V9.2.1-KnownPeople-UnknownFreq-CI4R"
    importlib.reload(_cfg)

    ci4r_payload, ci4r_cm = evaluate_one(
        Path(args.ci4r_run_dir),
        panel_label="(b) CI4R-MULTI3 only",
        slug="ci4r",
        out_dir=out_dir,
        top_k=args.top_k_pairs,
    )

    render_combined(
        out_dir / "fig2_strict_combined.pdf",
        [
            ("(a) Self (10/24 GHz + NRC augmented)", self_cm),
            ("(b) CI4R-MULTI3 only", ci4r_cm),
        ],
    )

    write_pair_check(
        {"self": self_payload, "ci4r": ci4r_payload},
        out_dir / "section_iiie_pair_check.md",
    )

    print(f"Self acc={self_payload['metrics']['accuracy']:.4f} F1={self_payload['metrics']['macro_f1']:.4f}")
    print(f"CI4R acc={ci4r_payload['metrics']['accuracy']:.4f} F1={ci4r_payload['metrics']['macro_f1']:.4f}")
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
