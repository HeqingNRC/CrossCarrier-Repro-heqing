"""Full public-baseline sweep driver: 8 native methods x 3 seeds = 24 runs, then eval.

Sequential (single GPU; the playbook says no parallel once compute-bound). Resumable:
any run whose pool_ep100_ema.pt already exists is skipped. Seed-major order so a complete
seed-42 row across all 8 methods lands first (a preliminary single-seed table early).
Reuse rows #9/#10/#11 are NOT here -- they are eval-only and already final.

Launch in background, redirect to a log; monitor with pb_monitor.py.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
SEEDS = [42, 1234, 31415]
GENERIC = ["vgg16_bn", "mobilenetv3_l", "effnetb0", "convnext_t", "swin_t", "convnextv2_t"]
EXTERNAL = ["radmamba", "selafd"]
EPOCHS = 100


def done(key, seed):
    return (HERE / "runs" / key / f"seed{seed}" / "checkpoints" / f"pool_ep{EPOCHS}_ema.pt").exists()


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
                print(f"[skip] {key} seed{seed} (pool_ep{EPOCHS}_ema.pt exists)", flush=True)
                n_skip += 1
                continue
            rd = HERE / "runs" / key / f"seed{seed}"
            if key in GENERIC:
                cmd = [PY, str(HERE / "pb_train_generic.py"), "--backbone", key,
                       "--seed", str(seed), "--epochs", str(EPOCHS), "--run-dir", str(rd)]
            else:
                cmd = [PY, str(HERE / "pb_external.py"), "--train", key,
                       "--seed", str(seed), "--epochs", str(EPOCHS), "--run-dir", str(rd)]
            rc = run(cmd)
            n_done += 1
            if rc != 0:
                print(f"[WARN] {key} seed{seed} exited {rc} -- continuing with the rest", flush=True)

    print(f"\n[sweep] training phase done: {n_done} ran, {n_skip} skipped, "
          f"{(time.time()-t0)/60:.1f} min", flush=True)
    print("[sweep] running unified evaluation (fills the table + latency + CIs)...", flush=True)
    run([PY, str(HERE / "pb_eval_unified.py"), "--latency"])
    print(f"[sweep] ALL DONE in {(time.time()-t0)/60:.1f} min. "
          f"See pb_results_auto.md + PUBLIC_BASELINE_RESULTS.md", flush=True)


if __name__ == "__main__":
    main()
