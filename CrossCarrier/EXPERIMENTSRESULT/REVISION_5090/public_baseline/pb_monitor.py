"""Live progress monitor for the public-baseline sweep (run in YOUR OWN terminal).

Scans the 24 expected runs (6 generic + 2 external) x 3 seeds, reads each run's
progress.json (current epoch, written every epoch) and final summary.json, and draws a
progress bar per run plus an overall bar with ETA. Pure stdlib -- no torch, instant.

Usage (PowerShell):
  & "C:\\Users\\Zirui Lin\\anaconda3\\envs\\Lider_5090\\python.exe" `
    "G:\\zhanghe\\Letter journal\\EXPERIMENTSRESULT\\REVISION_5090\\public_baseline\\pb_monitor.py" --watch
  # one-shot (no refresh): drop --watch ; change cadence: --interval 15
"""
import argparse
import re
import json
import os
import time
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
SWEEP_LOG = None
RESULTS_JSON = HERE / "pb_results.json"
SEEDS = [42, 1234, 31415]
GENERIC = ["vgg16_bn", "mobilenetv3_l", "effnetb0", "convnext_t", "swin_t", "convnextv2_t"]
EXTERNAL = ["radmamba", "selafd"]
ORDER = GENERIC + EXTERNAL
TOTAL_EP = 100
LABEL = {"vgg16_bn": "VGG16-BN", "mobilenetv3_l": "MobileNetV3-L", "effnetb0": "EfficientNet-B0",
         "convnext_t": "ConvNeXt-T", "swin_t": "Swin-T", "convnextv2_t": "ConvNeXtV2-T",
         "radmamba": "RadMamba", "selafd": "SelaFD"}


def latest_sweep_log():
    if SWEEP_LOG:
        p = Path(SWEEP_LOG)
        return p if p.exists() else None
    refs = HERE / "strong_sweep_latest_logs.txt"
    if not refs.exists():
        return None
    try:
        for line in refs.read_text(encoding="utf-8").splitlines():
            if line.startswith("stdout="):
                p = Path(line.split("=", 1)[1])
                return p if p.exists() else None
    except Exception:
        return None
    return None


def tail_lines(path, n=300):
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            return list(deque(f, maxlen=n))
    except Exception:
        return []


def parse_live_log():
    """Best-effort parse of the sweep stdout for the currently active run.

    This keeps monitoring useful for a child process that started before the richer
    progress.json writer was patched in. It only reads text; no model/GPU work.
    """
    p = latest_sweep_log()
    if not p:
        return None
    live = {"log": str(p)}
    for line in tail_lines(p):
        s = line.strip()
        if s.startswith("===") and ("pb_train_generic.py" in s or "pb_external.py" in s):
            kb = re.search(r"--backbone\s+([^\s]+)", s)
            kt = re.search(r"--train\s+([^\s]+)", s)
            sd = re.search(r"--seed\s+(\d+)", s)
            ep = re.search(r"--epochs\s+(\d+)", s)
            if kb or kt:
                live = {"log": str(p), "name": (kb or kt).group(1)}
                if sd:
                    live["seed"] = int(sd.group(1))
                if ep:
                    live["total"] = int(ep.group(1))
        m_setup = re.search(r"steps/epoch=(\d+)\s+total_steps=(\d+)", s)
        if m_setup:
            live["steps_per_epoch"] = int(m_setup.group(1))
            live["total_steps"] = int(m_setup.group(2))
        m_gen = re.search(
            r"ep\s+(\d+)/(\d+)\s+loss=([0-9.]+).*?"
            r"val_ema=([0-9.]+)/([0-9.]+)\s+test77_ema=([0-9.]+)/([0-9.]+).*?"
            r"lr_bb=([0-9.eE+-]+)\s+lr_head=([0-9.eE+-]+)", s)
        if m_gen:
            ep = int(m_gen.group(1)); total = int(m_gen.group(2))
            spe = int(live.get("steps_per_epoch", 40))
            live.update({
                "epoch": ep, "total": total, "step_in_epoch": spe,
                "global_step": ep * spe, "total_steps": total * spe,
                "train_loss": float(m_gen.group(3)),
                "val_acc_ema": float(m_gen.group(4)), "val_f1_ema": float(m_gen.group(5)),
                "test77_acc_ema": float(m_gen.group(6)), "test77_f1_ema": float(m_gen.group(7)),
                "lr_backbone": float(m_gen.group(8)), "lr_head": float(m_gen.group(9)),
                "last_log_line": s,
            })
        m_ext = re.search(
            r"ep\s+(\d+)/(\d+)\s+loss=([0-9.]+).*?"
            r"val_ema_f1=([0-9.]+)\s+test77_ema=([0-9.]+)/([0-9.]+).*?"
            r"lr=([0-9.eE+-]+)", s)
        if m_ext:
            ep = int(m_ext.group(1)); total = int(m_ext.group(2))
            spe = int(live.get("steps_per_epoch", 40))
            live.update({
                "epoch": ep, "total": total, "step_in_epoch": spe,
                "global_step": ep * spe, "total_steps": total * spe,
                "train_loss": float(m_ext.group(3)),
                "val_f1_ema": float(m_ext.group(4)),
                "test77_acc_ema": float(m_ext.group(5)), "test77_f1_ema": float(m_ext.group(6)),
                "lr": float(m_ext.group(7)),
                "last_log_line": s,
            })
    return live if "name" in live else None


def status(key, seed):
    d = RUNS / key / f"seed{seed}"
    ck = d / "checkpoints" / f"pool_ep{TOTAL_EP}_ema.pt"
    prog = d / "progress.json"
    summ = d / "reports" / "summary.json"
    if ck.exists():
        f1 = None
        if summ.exists():
            try:
                f1 = json.loads(summ.read_text(encoding="utf-8"))["final_ema"]["test77_macro_f1"]
            except Exception:
                pass
        return "done", TOTAL_EP, f1, (prog.stat().st_mtime if prog.exists() else ck.stat().st_mtime), {}
    if prog.exists():
        try:
            p = json.loads(prog.read_text(encoding="utf-8"))
            return "run", int(p.get("epoch", 0)), None, prog.stat().st_mtime, p
        except Exception:
            return "run", 0, None, prog.stat().st_mtime, {}
    return "wait", 0, None, None, {}


def bar(ep, total=TOTAL_EP, width=26):
    fr = (ep / total) if total else 0.0
    n = int(round(fr * width))
    return "#" * n + "-" * (width - n)


def fmt_eta(sec):
    if sec is None or sec <= 0 or sec != sec:
        return "--:--"
    h, r = divmod(int(sec), 3600)
    m, s = divmod(r, 60)
    return f"{h:d}h{m:02d}m" if h else f"{m:d}m{s:02d}s"


def fmt_float(x, digits=3):
    try:
        if x is None:
            return "-"
        return f"{float(x):.{digits}f}"
    except Exception:
        return "-"


def fmt_metrics(info):
    parts = []
    if info.get("global_step") is not None and info.get("total_steps"):
        parts.append(f"iter {info.get('global_step')}/{info.get('total_steps')}")
    elif info.get("step_in_epoch") is not None and info.get("steps_per_epoch"):
        parts.append(f"step {info.get('step_in_epoch')}/{info.get('steps_per_epoch')}")
    if info.get("train_loss") is not None:
        parts.append(f"loss={fmt_float(info.get('train_loss'), 4)}")
    if info.get("val_f1_ema") is not None:
        parts.append(f"srcF1={fmt_float(info.get('val_f1_ema'))}")
    if info.get("test77_f1_ema") is not None:
        acc = fmt_float(info.get("test77_acc_ema"))
        parts.append(f"77F1={fmt_float(info.get('test77_f1_ema'))}/acc={acc}")
    if info.get("gen_gap") is not None:
        parts.append(f"gap={float(info.get('gen_gap')):+.3f}")
    if info.get("lr_backbone") is not None:
        parts.append(f"lr={float(info.get('lr_backbone')):.1e}/{float(info.get('lr_head', 0.0)):.1e}")
    elif info.get("lr") is not None:
        parts.append(f"lr={float(info.get('lr')):.1e}")
    return "  " + " ".join(parts) if parts else ""


def render():
    rows = []
    mtimes = []
    ep_total = 0
    done_ct = 0
    live = parse_live_log()
    for k in ORDER:
        for s in SEEDS:
            st, ep, f1, mt, info = status(k, s)
            if live and st == "run" and live.get("name") == k and live.get("seed") == s:
                merged = dict(info)
                merged.update({kk: vv for kk, vv in live.items() if kk not in ("log", "last_log_line")})
                info = merged
                ep = int(info.get("epoch", ep))
            rows.append((k, s, st, ep, f1, info))
            ep_total += ep
            if mt:
                mtimes.append(mt)
            if st == "done":
                done_ct += 1
    n_runs = len(ORDER) * len(SEEDS)
    grand = n_runs * TOTAL_EP
    # ETA from observed throughput since the earliest progress write
    eta = None
    if mtimes and ep_total > 0:
        elapsed = time.time() - min(mtimes)
        if elapsed > 5 and ep_total < grand:
            rate = ep_total / elapsed                      # epochs/sec across the sweep
            eta = (grand - ep_total) / rate if rate > 0 else None

    lines = []
    lines.append("=" * 72)
    lines.append(f" Public-baseline sweep   runs {done_ct}/{n_runs} done   "
                 f"epochs {ep_total}/{grand} ({100*ep_total/grand:4.1f}%)   ETA {fmt_eta(eta)}")
    lines.append(" overall [" + bar(ep_total, grand, 50) + "]")
    lines.append("-" * 72)
    cur = None
    current = [r for r in rows if r[2] == "run"]
    if current:
        k, s, _st, ep, _f1, info = current[0]
        lines.append(f" Current: {LABEL[k]} seed{s} ep {ep}/{TOTAL_EP}{fmt_metrics(info)}")
    elif live:
        name = live.get("name", "?")
        lines.append(f" Current: {LABEL.get(name, name)} seed{live.get('seed', '?')} "
                     f"ep {live.get('epoch', '?')}/{live.get('total', TOTAL_EP)}{fmt_metrics(live)}")
    for k, s, st, ep, f1, info in rows:
        if k != cur:
            lines.append(f" {LABEL[k]}")
            cur = k
        tag = {"done": "OK ", "run": ">> ", "wait": " . "}[st]
        f1s = f"  F1={f1:.3f}" if (st == "done" and f1 is not None) else ""
        metrics = fmt_metrics(info) if st == "run" else ""
        lines.append(f"   {tag} seed{s:<6} [{bar(ep)}] {ep:3d}/{TOTAL_EP}{f1s}{metrics}")
    lines.append("-" * 72)
    if RESULTS_JSON.exists():
        lines.append(f" (table refreshed by pb_eval_unified -> {RESULTS_JSON.name})")
    lines.append(" Ctrl+C to stop monitoring (does NOT stop training).")
    return "\n".join(lines), done_ct == n_runs


def main():
    global RUNS, SWEEP_LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true", help="refresh until all runs are done")
    ap.add_argument("--interval", type=int, default=10, help="refresh seconds (default 10)")
    ap.add_argument("--runs-root", default=str(RUNS),
                    help="runs directory to monitor, e.g. public_baseline/runs_strong")
    ap.add_argument("--sweep-log", default=None,
                    help="optional sweep stdout log. If omitted, strong_sweep_latest_logs.txt is used when present.")
    args = ap.parse_args()
    RUNS = Path(args.runs_root)
    SWEEP_LOG = args.sweep_log
    while True:
        text, all_done = render()
        if args.watch:
            os.system("cls" if os.name == "nt" else "clear")
        print(text, flush=True)
        if not args.watch or all_done:
            break
        time.sleep(max(2, args.interval))


if __name__ == "__main__":
    main()
