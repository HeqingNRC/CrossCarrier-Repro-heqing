"""Driver for the generic public-baseline backbones (#1-6) x 3 seeds.

Modes:
  --dry-check         instantiate every backbone (downloads/loads timm weights) and
                      print trainable-param counts. NO training. Run this FIRST to
                      confirm all 6 timm checkpoints are available in the env.
  --smoke KEY         1-epoch smoke run of one backbone to a temp dir (pipeline sanity).
  (default)           full sweep: for each --backbones x --seeds, run pb_train_generic
                      unless pool_ep100_ema.pt already exists (resumable). Sequential.

The single-GPU playbook says generic baselines (no DAS -> no grid_sample) are a clean
GPU-cache job; run them sequentially (one ViT/CNN saturates the 5090 compute stage).

Examples:
  python pb_run.py --dry-check
  python pb_run.py --smoke convnext_t
  python pb_run.py --backbones convnext_t --seeds 42          # validate one method end-to-end
  python pb_run.py                                            # everything (6 backbones x 3 seeds)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pb_lib as P

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
PYEXE = sys.executable


def dry_check():
    import torch
    print(f"device={P.DEVICE}  torch={torch.__version__}")
    ok = True
    for key, name in P.GENERIC_BACKBONES.items():
        try:
            m = P.GenericBackbone(key)
            n = sum(p.numel() for p in m.parameters() if p.requires_grad)
            print(f"  OK  {key:14s} {name:42s} trainable={n/1e6:7.2f}M feat={m.feat_dim}")
            del m
            if P.DEVICE.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            ok = False
            print(f"  FAIL {key:14s} {name:42s} -> {type(e).__name__}: {e}")
    print("ALL BACKBONES OK" if ok else "SOME BACKBONES FAILED -- resolve before the sweep")
    return ok


def run_one(backbone, seed, epochs=100, extra=None):
    run_dir = RUNS / backbone / f"seed{seed}"
    done = run_dir / "checkpoints" / f"pool_ep{epochs}_ema.pt"
    if done.exists():
        print(f"[skip] {backbone} seed{seed} already done ({done})")
        return 0
    cmd = [PYEXE, str(HERE / "pb_train_generic.py"),
           "--backbone", backbone, "--seed", str(seed),
           "--epochs", str(epochs), "--run-dir", str(run_dir)]
    if extra:
        cmd += extra
    print("[run]", " ".join(cmd))
    return subprocess.call(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-check", action="store_true")
    ap.add_argument("--smoke", default=None, help="backbone key for a 1-epoch smoke test")
    ap.add_argument("--backbones", nargs="*", default=list(P.GENERIC_BACKBONES))
    ap.add_argument("--seeds", nargs="*", type=int, default=P.SEEDS)
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()

    if args.dry_check:
        sys.exit(0 if dry_check() else 1)
    if args.smoke:
        rd = RUNS / "_smoke" / args.smoke
        cmd = [PYEXE, str(HERE / "pb_train_generic.py"),
               "--backbone", args.smoke, "--seed", "42", "--epochs", "1",
               "--ema-start", "1", "--eval-every", "1", "--run-dir", str(rd)]
        print("[smoke]", " ".join(cmd))
        sys.exit(subprocess.call(cmd))

    rc = 0
    for bk in args.backbones:
        for s in args.seeds:
            rc = run_one(bk, s, epochs=args.epochs) or rc
    sys.exit(rc)


if __name__ == "__main__":
    main()
