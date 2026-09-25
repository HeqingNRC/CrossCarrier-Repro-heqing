"""Strong public-baseline sweep: optimized, fair recipes for rows #1-8.

This is intentionally separate from pb_sweep.py:
  - outputs go to runs_strong/ so the original plain runs stay reproducible;
  - generic rows use pb_train_generic.py --recipe strong;
  - external rows use pb_external.py --recipe strong;
  - checkpoint selection is still final EMA at ep100;
  - no DAS, no ACR, no carrier-ratio augmentation, no target/source-val selection.

After training it runs pb_eval_unified.py against runs_strong/ and writes:
  pb_results_strong.json
  pb_results_strong_auto.md
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
RUNS = HERE / "runs_strong"
SEEDS = [42, 1234, 31415]
GENERIC = ["vgg16_bn", "mobilenetv3_l", "effnetb0", "convnext_t", "swin_t", "convnextv2_t"]
EXTERNAL = ["radmamba", "selafd"]
EPOCHS = 100


def done(key, seed):
    return (RUNS / key / f"seed{seed}" / "checkpoints" / f"pool_ep{EPOCHS}_ema.pt").exists()


def run(cmd):
    print(f"\n=== [{time.strftime('%H:%M:%S')}] {' '.join(str(c) for c in cmd[1:])} ===", flush=True)
    rc = subprocess.call(cmd)
    print(f"=== exit {rc} ===", flush=True)
    return rc


def main():
    t0 = time.time()
    n_done = n_skip = 0
    for seed in SEEDS:
        for key in GENERIC + EXTERNAL:
            if done(key, seed):
                print(f"[skip] strong {key} seed{seed} (pool_ep{EPOCHS}_ema.pt exists)", flush=True)
                n_skip += 1
                continue
            rd = RUNS / key / f"seed{seed}"
            if key in GENERIC:
                cmd = [PY, str(HERE / "pb_train_generic.py"), "--recipe", "strong",
                       "--backbone", key, "--seed", str(seed), "--epochs", str(EPOCHS),
                       "--run-dir", str(rd)]
            else:
                cmd = [PY, str(HERE / "pb_external.py"), "--recipe", "strong",
                       "--train", key, "--seed", str(seed), "--epochs", str(EPOCHS),
                       "--run-dir", str(rd)]
            rc = run(cmd)
            n_done += 1
            if rc != 0:
                print(f"[WARN] strong {key} seed{seed} exited {rc} -- continuing with the rest", flush=True)

    print(f"\n[strong-sweep] training phase done: {n_done} ran, {n_skip} skipped, "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)
    print("[strong-sweep] running unified evaluation against runs_strong/ ...", flush=True)
    run([PY, str(HERE / "pb_eval_unified.py"),
         "--runs-root", str(RUNS),
         "--out-prefix", "pb_results_strong",
         "--latency"])
    print(f"[strong-sweep] ALL DONE in {(time.time() - t0) / 60:.1f} min. "
          "See pb_results_strong_auto.md", flush=True)


if __name__ == "__main__":
    main()
