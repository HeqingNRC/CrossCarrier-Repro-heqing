"""Cross-dataset evaluation: V9.2.1 best checkpoint -> CI4R 7-class.

Loads a V9.2.1 EMA / best / last checkpoint and evaluates it on the
public CI4R dataset (downloaded under public_dataset/ci4r/data/) using
the V9.2.1 7-class label space. Reports per-frequency and overall
accuracy / macro-F1, and writes a JSON report next to the run's other
reports.

Usage
-----
    $PY = 'C:\\Users\\Zirui Lin\\anaconda3\\envs\\z\\python.exe'
    & $PY G:\\zhanghe\\baseline_v9_2_1\\cross_dataset_eval.py `
        --run-dir  G:\\zhanghe\\output\\V9.2.1-KnownPeople-UnknownFreq\\run_001 `
        --ckpt     ema

Outputs
-------
    <run-dir>/reports/cross_dataset_ci4r_<ckpt>.json
    <run-dir>/reports/cross_dataset_ci4r_<ckpt>.md
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

import config
from v9_2_1lib import (
    ModelEMA,
    TimmBackboneV921,
    evaluate,
    make_eval_loader,
)


CI4R_MANIFEST = Path("G:/zhanghe/public_dataset/manifests/ci4r_7c.csv")
V921_FREQS = ["10GHz", "24GHz", "77GHz"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True,
                    help="completed V9.2.1 run dir containing checkpoints/")
    ap.add_argument("--ckpt", choices=["ema", "best", "last"], default="ema")
    ap.add_argument("--backbone", default=config.DEFAULT_BACKBONE)
    ap.add_argument("--adapter-mode", choices=["lora", "last_block"],
                    default=None,
                    help="default: read from checkpoint metadata; falls back to config.BACKBONE_TUNE_MODE")
    ap.add_argument("--manifest", default=str(CI4R_MANIFEST))
    return ap.parse_args()


def amp_dtype():
    return torch.bfloat16 if config.AMP_DTYPE == "bf16" else torch.float16


def safe_name(backbone):
    return backbone.replace("/", "_").replace(".", "_").replace(":", "_")


def resolve_ckpt(run_dir: Path, backbone: str, ckpt: str) -> tuple[Path, str]:
    if ckpt == "ema":
        return run_dir / "checkpoints" / f"{safe_name(backbone)}_best_ema.pt", "ema"
    if ckpt == "best":
        return run_dir / "checkpoints" / f"{safe_name(backbone)}_best_live.pt", "live"
    return run_dir / "checkpoints" / f"{safe_name(backbone)}_last.pt", "last"


def load_ckpt(model, ema, path: Path, selector: str, device):
    ckpt = torch.load(path, map_location=device)
    meta = {
        "epoch": ckpt.get("epoch"),
        "adapter_mode": ckpt.get("adapter_mode"),
        "backbone": ckpt.get("backbone", config.DEFAULT_BACKBONE),
        "selection_metric": ckpt.get("selection_metric"),
        "dev77_f1": ckpt.get("dev77_f1"),
    }
    if selector == "ema":
        ema.load_state_dict(ckpt["ema"])
        ema.copy_to(model)
    elif selector == "live":
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt["model"])
        if "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
    return meta


def load_ci4r_df(manifest_path: Path) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)
    df["class_idx_7c"] = df["class_idx"].astype(int)
    df["freq_idx"] = -1
    df = df[df["frequency"].isin(V921_FREQS)].reset_index(drop=True)
    return df


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise SystemExit(f"run_dir not found: {run_dir}")
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    ckpt_path, selector = resolve_ckpt(run_dir, args.backbone, args.ckpt)
    if not ckpt_path.exists() and args.ckpt == "ema":
        fallback = run_dir / "checkpoints" / f"{safe_name(args.backbone)}_best_live.pt"
        if fallback.exists():
            print(f"[fallback] {ckpt_path.name} missing -> {fallback.name}")
            ckpt_path, selector = fallback, "live"
        else:
            fallback = run_dir / "checkpoints" / f"{safe_name(args.backbone)}_last.pt"
            print(f"[fallback] -> {fallback.name}")
            ckpt_path, selector = fallback, "last"
    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")

    head = torch.load(ckpt_path, map_location="cpu")
    adapter_mode = args.adapter_mode or head.get("adapter_mode") or config.BACKBONE_TUNE_MODE
    backbone_name = head.get("backbone", args.backbone)
    print(f"ckpt={ckpt_path.name} adapter_mode={adapter_mode} backbone={backbone_name}")

    model = TimmBackboneV921(
        backbone_name, config.NUM_CLASSES, adapter_mode=adapter_mode
    ).to(device)
    ema = ModelEMA(model, config.EMA_DECAY)

    meta = load_ckpt(model, ema, ckpt_path, selector, device)
    model.eval()
    print(f"loaded epoch={meta['epoch']} dev77_f1={meta['dev77_f1']}")

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise SystemExit(f"CI4R manifest not found: {manifest_path}; "
                         "run public_dataset/ingest/ci4r_make_manifest.py first.")
    df = load_ci4r_df(manifest_path)
    print(f"CI4R 7c: {len(df)} rows total")

    per_freq = {}
    overall_correct = overall_total = 0
    for freq in V921_FREQS:
        df_f = df[df["frequency"] == freq].reset_index(drop=True)
        if df_f.empty:
            continue
        loader = make_eval_loader(df_f)
        r = evaluate(model, loader, device, amp_dtype())
        per_freq[freq] = {
            "n_samples": int(len(df_f)),
            "acc": float(r["acc"]),
            "macro_f1": float(r["macro_f1"]),
            "per_class": {k: float(v) for k, v in r["per_class"].items()},
        }
        overall_correct += int(round(r["acc"] * len(df_f)))
        overall_total += int(len(df_f))
        print(f"  [{freq}] n={len(df_f):4d}  acc={r['acc']:.4f}  f1={r['macro_f1']:.4f}")

    overall_acc = overall_correct / max(1, overall_total)
    print(f"  [overall] n={overall_total} acc(weighted-avg)={overall_acc:.4f}")

    summary = {
        "task": "cross_dataset_eval_self_to_ci4r_7class",
        "checkpoint": ckpt_path.name,
        "ckpt_selector": args.ckpt,
        "backbone": backbone_name,
        "adapter_mode": adapter_mode,
        "source_run": str(run_dir),
        "target_dataset": "CI4R-MULTI3 (Gurbuz lab UAlabama, public)",
        "target_manifest": str(manifest_path),
        "source_dev77_f1_at_ckpt": meta.get("dev77_f1"),
        "source_epoch_at_ckpt": meta.get("epoch"),
        "per_frequency": per_freq,
        "overall_acc_weighted": overall_acc,
        "n_samples_overall": overall_total,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    json_out = reports_dir / f"cross_dataset_ci4r_{args.ckpt}.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"saved {json_out}")

    md_out = reports_dir / f"cross_dataset_ci4r_{args.ckpt}.md"
    lines = []
    lines.append(f"# Cross-dataset eval: V9.2.1 -> CI4R (7-class)")
    lines.append("")
    lines.append(f"- Generated: {summary['generated_at']}")
    lines.append(f"- Source run: `{summary['source_run']}`")
    lines.append(f"- Checkpoint: `{summary['checkpoint']}` ({summary['ckpt_selector']})")
    lines.append(f"- Adapter: `{adapter_mode}`")
    lines.append(f"- Source 77GHz dev macro-F1 at ckpt: {summary['source_dev77_f1_at_ckpt']}")
    lines.append(f"- CI4R 7-class manifest: `{manifest_path.name}` ({overall_total} samples)")
    lines.append("")
    lines.append("## Per-frequency results on CI4R")
    lines.append("")
    lines.append("| frequency | n | accuracy | macro-F1 |")
    lines.append("|---|---:|---:|---:|")
    for freq, r in per_freq.items():
        lines.append(f"| {freq} | {r['n_samples']} | {r['acc']:.4f} | {r['macro_f1']:.4f} |")
    lines.append(f"| **overall (weighted-avg acc)** | **{overall_total}** | **{overall_acc:.4f}** | -- |")
    lines.append("")
    lines.append("## Per-class breakdown (per frequency)")
    lines.append("")
    for freq, r in per_freq.items():
        lines.append(f"### {freq}")
        lines.append("")
        lines.append("| class | F1 |")
        lines.append("|---|---:|")
        for c, v in r["per_class"].items():
            lines.append(f"| {c} | {v:.4f} |")
        lines.append("")

    with open(md_out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")
    print(f"saved {md_out}")


if __name__ == "__main__":
    main()
