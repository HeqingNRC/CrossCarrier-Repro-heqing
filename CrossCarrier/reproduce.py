#!/usr/bin/env python
"""One-command reproduction of the headline system.

Runs the two audited eval scripts, in order, from the repo root:

  1) baseline_v20/ensemble_rules.py
       -> single-seed macro-F1 (3 seeds) + the THREE ensemble rules
          (majority / logit-avg / posterior) on the full 418-image 77 GHz target set.
          This is where the headline 0.832 (single) and 0.857 (logit-avg ensemble) come from.

  2) EXPERIMENTSRESULT/REVISION_5090/public_baseline/pb_eval_unified.py --only A_V13_GRL
       -> the same numbers through the unified public-baseline harness, plus the
          bootstrap 95% CI of the deployment ensemble row (#11).

Nothing is trained: both scripts are eval-only over the shipped final-EMA checkpoints.
Expected wall-clock: a few minutes on a CUDA GPU (plus a one-off ~1.2 GB DINOv3
backbone download from the HuggingFace Hub on the very first run).

Usage:
    python reproduce.py              # both steps
    python reproduce.py --step 1     # ensemble rules only
    python reproduce.py --step 2     # unified harness + bootstrap CI only
    python reproduce.py --n-boot 10000
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# The published values, printed next to the recomputed ones so a mismatch is obvious.
EXPECTED = """
  Proposed (A_V13_GRL), full-418 77 GHz, final-EMA, seeds 42/1234/31415
    single macro-F1 .............. 0.832 +/- 0.034   (per-seed 0.791 / 0.832 / 0.874)
    ensemble, majority vote ...... 0.851
    ensemble, logit-average ...... 0.857   <-- HEADLINE (paper's deployment number)
    ensemble, posterior-average .. 0.856   boot95 CI [0.821, 0.889]
"""


BACKBONE_CACHE = (ROOT / "weights" / "hub"
                  / "models--timm--vit_large_patch16_dinov3.lvd1689m")


def build_env() -> dict:
    """Child env, with the HuggingFace offline latch set to match what we ship.

    `baseline_v8/v8lib.py` points HF_HOME / HUGGINGFACE_HUB_CACHE at ./weights and does
    os.environ.setdefault('HF_HUB_OFFLINE', '1'). Since this bundle ships the DINOv3
    backbone inside weights/hub, the default offline mode is exactly right -- no network
    is touched. Only if that cache is missing do we release the latch so timm can
    download it (presetting the variable wins over setdefault()).
    """
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["HF_HOME"] = str(ROOT / "weights")
    env["HUGGINGFACE_HUB_CACHE"] = str(ROOT / "weights" / "hub")
    if BACKBONE_CACHE.is_dir():
        env["HF_HUB_OFFLINE"] = "1"          # bundled backbone -> stay fully offline
        print(f"[env] bundled DINOv3 backbone found -> offline mode "
              f"({BACKBONE_CACHE.relative_to(ROOT)})")
    else:
        env["HF_HUB_OFFLINE"] = "0"
        env["TRANSFORMERS_OFFLINE"] = "0"
        print("[env] bundled backbone NOT found -> online mode; timm will download "
              "vit_large_patch16_dinov3.lvd1689m (~1.2 GB) into ./weights/hub on first use")
    return env


def run(script: Path, args: list[str], env: dict) -> int:
    cmd = [sys.executable, str(script), *args]
    print(f"\n{'=' * 78}\n$ {' '.join(cmd)}\n{'=' * 78}", flush=True)
    # cwd=ROOT matters: both scripts resolve EXPERIMENTSRESULT/... relative to the CWD.
    return subprocess.call(cmd, cwd=str(ROOT), env=env)


def preflight() -> None:
    missing = []
    for seed in (42, 1234, 31415):
        ck = (ROOT / "EXPERIMENTSRESULT" / "REVISION_5090" / "A_V13_GRL"
              / f"seed{seed}" / "checkpoints" / "pool_ep100_ema.pt")
        if not ck.exists():
            missing.append(str(ck.relative_to(ROOT)))
    if not (ROOT / "tasks" / "known_people_unknown_freq" / "manifest" / "test.csv").exists():
        missing.append("tasks/known_people_unknown_freq/manifest/test.csv")
    if missing:
        print("[FATAL] missing required files:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        sys.exit(2)

    try:
        import torch
    except Exception as exc:                                  # noqa: BLE001
        print(f"[FATAL] cannot import torch: {exc}\n"
              f"        install the dependencies first: pip install -r requirements.txt",
              file=sys.stderr)
        sys.exit(2)
    if not torch.cuda.is_available():
        print("[WARN] no CUDA device visible -- the scripts will fall back to CPU and be "
              "very slow. The published numbers were produced on an RTX 5090.")
    else:
        print(f"[env] torch {torch.__version__}  |  GPU {torch.cuda.get_device_name(0)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, choices=(1, 2), default=None,
                    help="run only step 1 or only step 2 (default: both)")
    ap.add_argument("--n-boot", type=int, default=2000,
                    help="bootstrap resamples for the CI in step 2 (published: 2000)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    ap.add_argument("--output-root", type=Path, default=ROOT / "output" / "reproduction")
    args = ap.parse_args()
    if args.batch_size < 1 or args.n_boot < 1:
        ap.error("batch size and bootstrap count must be positive")

    print(__doc__.split("Usage:")[0].rstrip())
    print("Published values for comparison:")
    print(EXPECTED)

    preflight()
    env = build_env()
    env["XC_EVAL_BATCH"] = str(args.batch_size)
    env["XC_EVAL_PRECISION"] = args.precision
    env["XC_REPRO_OUTPUT"] = str(args.output_root.resolve())
    args.output_root.mkdir(parents=True, exist_ok=True)
    import json
    import platform
    import torch
    from importlib.metadata import version, PackageNotFoundError
    versions = {}
    for name in ("torch", "torchvision", "timm", "peft", "transformers", "numpy", "pillow", "huggingface_hub"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    metadata = {"python": platform.python_version(), "packages": versions, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "batch_size": args.batch_size, "requested_precision": args.precision, "protocol": "legacy full-418, shipped final-EMA checkpoints", "note": "New inference run; numerical differences possible across hardware and precision."}
    (args.output_root / "environment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[output] {args.output_root.resolve()}", flush=True)
    rc = 0

    if args.step in (None, 1):
        rc |= run(ROOT / "baseline_v20" / "ensemble_rules.py", [], env)
        print(f"\n[step 1] exit={rc}, outputs: {args.output_root / 'ensemble'}")
        print("[note] 'DAS only (A_REF)' and 'DAS jitter (E1_jitter)' print [skip]: those "
              "two ablation checkpoints are NOT shipped (see README, Known limitations). "
              "The proposed row is unaffected.")

    if args.step in (None, 2):
        rc |= run(ROOT / "EXPERIMENTSRESULT" / "REVISION_5090" / "public_baseline"
                  / "pb_eval_unified.py",
                  ["--only", "A_V13_GRL", "--n-boot", str(args.n_boot)], env)
        print(f"\n[step 2] cumulative exit={rc}, outputs: {args.output_root / 'unified'}")
        print("[note] rows #1-#9 print NOT RUN: only the proposed checkpoints are shipped. "
              "Their published values are in docs/pb_results_strong_auto.md.")

    print("\nPublished values, again, for comparison:")
    print(EXPECTED)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
