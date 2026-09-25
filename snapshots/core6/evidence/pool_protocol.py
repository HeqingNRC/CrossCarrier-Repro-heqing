"""Candidate-pool selection protocol for V10-V18 (160-epoch reruns).

Protocol
--------
Every `period` epochs (default 10) check the EMA model's source-val accuracy.
If acc >= `threshold` (default 0.90), the EMA snapshot becomes a pool member:
  * checkpoint persisted to `<run_dir>/checkpoints/pool_ep{NNN}_ema.pt`
  * 77 GHz dev + final-test evaluated and recorded
  * pool_log.json/md updated incrementally
At training end, the pool member with the strongest 77 GHz dev macro-F1 is
copied to `<run_dir>/checkpoints/best_pool_ema.pt` and a markdown summary is
written to `<run_dir>/reports/pool_summary.md`.

This module is dependency-light: callers only need to provide an `evaluate`
function (with the same signature as v9_2_1lib.evaluate), eval loaders, and
the EMA wrapper they already use. The caller is responsible for being inside
its own SwapEMA context when invoking `maybe_update_pool` so that the model
holds EMA weights at the time of evaluation.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


_DEFAULT_THRESHOLD = float(os.environ.get("POOL_THRESHOLD", "0.90"))
_DEFAULT_PERIOD = int(os.environ.get("POOL_PERIOD", "10"))


def init_pool(
    run_dir: Path,
    *,
    threshold: float = _DEFAULT_THRESHOLD,
    period: int = _DEFAULT_PERIOD,
    selection_metric: str = "dev77_f1",
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Create pool state. Call once before the epoch loop."""
    run_dir = Path(run_dir)
    pool_ckpt_dir = run_dir / "checkpoints"
    pool_ckpt_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    state: Dict[str, Any] = {
        "run_dir": run_dir,
        "ckpt_dir": pool_ckpt_dir,
        "reports_dir": reports_dir,
        "threshold": float(threshold),
        "period": int(period),
        "selection_metric": selection_metric,
        "members": [],
        "best_idx": -1,
        "log_fn": log_fn or print,
        "json_path": reports_dir / "pool_log.json",
        "md_path": reports_dir / "pool_log.md",
        "summary_path": reports_dir / "pool_summary.md",
        "summary_json_path": reports_dir / "pool_summary.json",
    }
    _write_json(state)
    _write_md(state)
    return state


def maybe_update_pool(
    state: Dict[str, Any],
    *,
    epoch: int,
    total_epochs: int,
    val_ema_metrics: Dict[str, float],
    model: Any,
    ema: Any,
    backbone: str,
    evaluate_fn: Callable,
    dev77_loader: Any,
    test77_final_loader: Any,
    device: Any,
    amp_dtype: Any,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> bool:
    """Per-epoch hook. Returns True if a member was added this epoch.

    Caller MUST already be inside its SwapEMA(model, ema) context so that
    `model` carries EMA weights when `evaluate_fn` is invoked. After this
    function returns the SwapEMA context will restore live weights as usual.
    """
    period = state["period"]
    if epoch <= 0 or epoch % period != 0:
        return False

    val_acc = float(val_ema_metrics.get("acc", 0.0))
    val_f1 = float(val_ema_metrics.get("macro_f1", 0.0))
    threshold = state["threshold"]
    log = state["log_fn"]

    log(
        f"[pool] check ep={epoch:03d}/{total_epochs} "
        f"source_val_acc_ema={val_acc:.4f} threshold={threshold:.2f}"
    )

    if val_acc < threshold:
        log(f"[pool] ep={epoch:03d} SKIP (source_val_acc_ema={val_acc:.4f} < {threshold:.2f})")
        return False

    # Eligible -> evaluate on 77 GHz dev + final under the *current* (EMA) weights.
    dev77 = evaluate_fn(model, dev77_loader, device, amp_dtype)
    final77 = evaluate_fn(model, test77_final_loader, device, amp_dtype)

    ckpt_name = f"pool_ep{epoch:03d}_ema.pt"
    ckpt_path = state["ckpt_dir"] / ckpt_name
    payload = {
        "ema": ema.state_dict(),
        "backbone": backbone,
        "epoch": epoch,
        "selection_metric": state["selection_metric"],
        "source_val_acc_ema": val_acc,
        "source_val_f1_ema": val_f1,
        "dev77_acc": float(dev77["acc"]),
        "dev77_f1": float(dev77["macro_f1"]),
        "dev77_per_class": dev77.get("per_class", {}),
        "test77_acc": float(final77["acc"]),
        "test77_f1": float(final77["macro_f1"]),
        "test77_per_class": final77.get("per_class", {}),
    }
    if extra_meta:
        payload.update({k: v for k, v in extra_meta.items() if k not in payload})
    _atomic_torch_save(payload, ckpt_path)

    member = {
        "epoch": int(epoch),
        "ckpt_path": str(ckpt_path),
        "source_val_acc_ema": val_acc,
        "source_val_f1_ema": val_f1,
        "dev77_acc": float(dev77["acc"]),
        "dev77_f1": float(dev77["macro_f1"]),
        "dev77_per_class": dev77.get("per_class", {}),
        "test77_acc": float(final77["acc"]),
        "test77_f1": float(final77["macro_f1"]),
        "test77_per_class": final77.get("per_class", {}),
    }
    state["members"].append(member)
    _recompute_best(state)
    _write_json(state)
    _write_md(state)

    best = state["members"][state["best_idx"]]
    log(
        f"[pool] ep={epoch:03d} ADDED "
        f"src_val={val_acc:.4f}/{val_f1:.4f} "
        f"dev77={member['dev77_acc']:.4f}/{member['dev77_f1']:.4f} "
        f"final77={member['test77_acc']:.4f}/{member['test77_f1']:.4f}"
    )
    log(
        f"[pool] running best ep={best['epoch']:03d} "
        f"dev77_f1={best['dev77_f1']:.4f} final77_f1={best['test77_f1']:.4f} "
        f"(pool_size={len(state['members'])})"
    )
    return True


def finalize_pool(state: Dict[str, Any]) -> Dict[str, Any]:
    """Call once after the epoch loop. Writes summary and copies the winner."""
    log = state["log_fn"]
    members = state["members"]
    if not members:
        log("[pool] FINAL: no member crossed the source-val threshold; pool is empty")
        summary = {
            "pool_threshold": state["threshold"],
            "pool_period": state["period"],
            "selection_metric": state["selection_metric"],
            "pool_size": 0,
            "members": [],
            "best": None,
        }
        with open(state["summary_json_path"], "w") as f:
            json.dump(summary, f, indent=2)
        with open(state["summary_path"], "w") as f:
            f.write(_summary_md(state, summary))
        return summary

    _recompute_best(state)
    best_idx = state["best_idx"]
    best = members[best_idx]

    best_copy = state["ckpt_dir"] / "best_pool_ema.pt"
    shutil.copy2(best["ckpt_path"], best_copy)

    summary = {
        "pool_threshold": state["threshold"],
        "pool_period": state["period"],
        "selection_metric": state["selection_metric"],
        "pool_size": len(members),
        "members": members,
        "best": {**best, "best_pool_ema_path": str(best_copy)},
    }
    with open(state["summary_json_path"], "w") as f:
        json.dump(summary, f, indent=2)
    with open(state["summary_path"], "w") as f:
        f.write(_summary_md(state, summary))

    log(
        f"[pool] FINAL pool_size={len(members)} "
        f"best ep={best['epoch']:03d} "
        f"dev77_f1={best['dev77_f1']:.4f} final77_f1={best['test77_f1']:.4f} "
        f"-> {best_copy}"
    )
    return summary


def _recompute_best(state: Dict[str, Any]) -> None:
    metric = state["selection_metric"]
    if metric == "test77_f1":
        key = "test77_f1"
    else:
        key = "dev77_f1"
    best_idx = -1
    best_val = -1.0
    for i, m in enumerate(state["members"]):
        v = float(m.get(key, -1.0))
        if v > best_val:
            best_val = v
            best_idx = i
    state["best_idx"] = best_idx


def _atomic_torch_save(obj: Any, path: Path) -> None:
    import torch
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _write_json(state: Dict[str, Any]) -> None:
    payload = {
        "pool_threshold": state["threshold"],
        "pool_period": state["period"],
        "selection_metric": state["selection_metric"],
        "members": state["members"],
        "best_idx": state["best_idx"],
    }
    with open(state["json_path"], "w") as f:
        json.dump(payload, f, indent=2)


def _write_md(state: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# Pool log (live)")
    lines.append("")
    lines.append(f"- threshold: source_val_acc_ema >= {state['threshold']:.2f}")
    lines.append(f"- period: every {state['period']} epochs")
    lines.append(f"- selection metric: {state['selection_metric']}")
    lines.append("")
    if not state["members"]:
        lines.append("_no member yet._")
    else:
        lines.append("| epoch | src_val_acc | src_val_F1 | dev77 acc | dev77 F1 | final77 acc | final77 F1 |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|")
        for m in state["members"]:
            lines.append(
                f"| {m['epoch']:3d} | {m['source_val_acc_ema']:.4f} | {m['source_val_f1_ema']:.4f} "
                f"| {m['dev77_acc']:.4f} | {m['dev77_f1']:.4f} "
                f"| {m['test77_acc']:.4f} | {m['test77_f1']:.4f} |"
            )
        if state["best_idx"] >= 0:
            best = state["members"][state["best_idx"]]
            lines.append("")
            lines.append(
                f"**running best** by `{state['selection_metric']}`: "
                f"epoch {best['epoch']}, dev77 F1 {best['dev77_f1']:.4f}, "
                f"final77 F1 {best['test77_f1']:.4f}"
            )
    with open(state["md_path"], "w") as f:
        f.write("\n".join(lines) + "\n")


def _summary_md(state: Dict[str, Any], summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Pool selection summary")
    lines.append("")
    lines.append(f"- threshold: source_val_acc_ema >= {summary['pool_threshold']:.2f}")
    lines.append(f"- period: every {summary['pool_period']} epochs")
    lines.append(f"- pool size: {summary['pool_size']}")
    lines.append(f"- selection metric: {summary['selection_metric']}")
    lines.append("")
    if summary["pool_size"] == 0:
        lines.append("No checkpoint crossed the source-val threshold during training.")
        return "\n".join(lines) + "\n"
    lines.append("## Members")
    lines.append("")
    lines.append("| epoch | src_val_acc | src_val_F1 | dev77 acc | dev77 F1 | final77 acc | final77 F1 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for m in summary["members"]:
        lines.append(
            f"| {m['epoch']:3d} | {m['source_val_acc_ema']:.4f} | {m['source_val_f1_ema']:.4f} "
            f"| {m['dev77_acc']:.4f} | {m['dev77_f1']:.4f} "
            f"| {m['test77_acc']:.4f} | {m['test77_f1']:.4f} |"
        )
    lines.append("")
    best = summary["best"]
    lines.append("## Selected (best in pool)")
    lines.append("")
    lines.append(f"- epoch: **{best['epoch']}**")
    lines.append(f"- source val acc / F1 (EMA): {best['source_val_acc_ema']:.4f} / {best['source_val_f1_ema']:.4f}")
    lines.append(f"- 77GHz dev   acc / F1: {best['dev77_acc']:.4f} / {best['dev77_f1']:.4f}")
    lines.append(f"- 77GHz final acc / F1: {best['test77_acc']:.4f} / {best['test77_f1']:.4f}")
    lines.append(f"- checkpoint: `{best['best_pool_ema_path']}`")
    if best.get("dev77_per_class"):
        lines.append("")
        lines.append("### dev77 per-class F1")
        for k, v in best["dev77_per_class"].items():
            lines.append(f"- {k}: {float(v):.4f}")
    if best.get("test77_per_class"):
        lines.append("")
        lines.append("### final77 per-class F1")
        for k, v in best["test77_per_class"].items():
            lines.append(f"- {k}: {float(v):.4f}")
    return "\n".join(lines) + "\n"
