"""P1 : paired stratified bootstrap 95% CI for the key deltas (NO training).

Reuses eval_out/logits_cache.npz produced by ensemble_rules.py (per-seed final-EMA
logits over the 418 target clips). Resampling is PAIRED (same resampled clip indices
for both sides of a delta) and STRATIFIED by true class (per-class counts preserved).

Deltas (macro-F1, full-418):
  1. DAS+ACR single  -  DAS single        : A_V13_GRL vs A_REF   (seed-averaged)
  2. DAS single      -  jitter single     : A_REF     vs E1_jitter (seed-averaged)
  3. DAS+ACR ensemble - DAS+ACR single    : A_V13_GRL posterior-avg vs its seed mean
  (Proposed - best EXTERNAL baseline needs the public-baseline per-clip predictions;
   wired as a TODO below -- drop a logits/preds file in eval_out to enable it.)

"single" macro-F1 on a resample = mean over the available seeds of that config's
per-seed macro-F1 on the resampled clips. "ensemble" = macro-F1 of the posterior-avg
prediction vector on the resampled clips.

CI = 2.5/97.5 percentiles of the bootstrap delta distribution; also reports the
observed (full-sample) delta and the fraction of resamples with delta>0.

Deterministic: numpy default_rng(SEED). Outputs labelled PRELIMINARY.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import config

NC = config.NUM_CLASSES
REV = Path("EXPERIMENTSRESULT/REVISION_5090")
OUT = REV / "SUPP_ABLATION" / "eval_out"
NPZ = OUT / "logits_cache.npz"
B = 10000
SEED = 0


def macro_f1(preds, y, idx):
    p = preds[idx]; t = y[idx]
    f1s = []
    for c in range(NC):
        tp = int(((p == c) & (t == c)).sum())
        fp = int(((p == c) & (t != c)).sum())
        fn = int(((p != c) & (t == c)).sum())
        pr = tp / (tp + fp) if (tp + fp) else 0.0
        rc = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if (pr + rc) else 0.0)
    return float(np.mean(f1s))


def single_f1(L, y, idx):
    """seed-averaged single-model macro-F1 on resample idx. L: [S, N, NC]."""
    preds = L.argmax(2)                       # [S, N]
    return float(np.mean([macro_f1(preds[s], y, idx) for s in range(L.shape[0])]))


def posterior_pred(L):
    e = np.exp(L - L.max(axis=2, keepdims=True))
    post = (e / e.sum(axis=2, keepdims=True)).mean(0)   # [N, NC]
    return post.argmax(1)


def strat_resampler(y, rng):
    by_class = [np.where(y == c)[0] for c in range(NC)]

    def draw():
        return np.concatenate([
            rng.choice(ix, size=len(ix), replace=True) for ix in by_class if len(ix)
        ])
    return draw


def run_delta(name, fn_a, fn_b, y, draw, rng):
    full_idx = np.arange(len(y))
    obs = fn_a(full_idx) - fn_b(full_idx)
    deltas = np.empty(B)
    for b in range(B):
        idx = draw()
        deltas[b] = fn_a(idx) - fn_b(idx)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    frac_pos = float((deltas > 0).mean())
    rec = {"delta": name, "observed_pp": round(obs * 100, 2),
           "ci95_pp": [round(lo * 100, 2), round(hi * 100, 2)],
           "frac_gt0": round(frac_pos, 4), "boot_mean_pp": round(float(deltas.mean()) * 100, 2)}
    print(f"{name:38s} obs={obs*100:+.2f}pp  95%CI=[{lo*100:+.2f},{hi*100:+.2f}]pp  "
          f"P(>0)={frac_pos:.3f}")
    return rec


def main():
    if not NPZ.exists():
        raise SystemExit(f"missing {NPZ}; run ensemble_rules.py first.")
    z = np.load(NPZ, allow_pickle=True)
    y = z["y"]
    have = [k for k in z.files if k not in ("y", "classes")]
    print(f"[data] N={len(y)} configs in cache: {have}")
    rng = np.random.default_rng(SEED)
    draw = strat_resampler(y, rng)

    recs = []
    if "A_V13_GRL" in have and "A_REF" in have:
        L_grl, L_ref = z["A_V13_GRL"], z["A_REF"]
        recs.append(run_delta(
            "DAS+ACR single - DAS single",
            lambda i: single_f1(L_grl, y, i), lambda i: single_f1(L_ref, y, i),
            y, draw, rng))
    if "A_REF" in have and "E1_jitter" in have:
        L_ref, L_jit = z["A_REF"], z["E1_jitter"]
        recs.append(run_delta(
            "DAS single - jitter single",
            lambda i: single_f1(L_ref, y, i), lambda i: single_f1(L_jit, y, i),
            y, draw, rng))
    if "A_V13_GRL" in have:
        L_grl = z["A_V13_GRL"]
        ens = posterior_pred(L_grl)
        recs.append(run_delta(
            "DAS+ACR ensemble - DAS+ACR single",
            lambda i: macro_f1(ens, y, i), lambda i: single_f1(L_grl, y, i),
            y, draw, rng))

    # TODO(external): to enable "Proposed - best external baseline", save the
    # baseline's per-clip argmax preds (aligned to the 418 manifest order) as
    # eval_out/ext_baseline_preds.npy, then add a run_delta with macro_f1(ext, ...).
    ext = OUT / "ext_baseline_preds.npy"
    if ext.exists() and "A_V13_GRL" in have:
        ext_pred = np.load(ext)
        recs.append(run_delta(
            "Proposed ensemble - external baseline",
            lambda i: macro_f1(posterior_pred(z["A_V13_GRL"]), y, i),
            lambda i: macro_f1(ext_pred, y, i), y, draw, rng))
    else:
        print("[note] external-baseline delta skipped (no eval_out/ext_baseline_preds.npy)")

    out = {"n_clips": int(len(y)), "n_boot": B, "rng_seed": SEED,
           "note": "PRELIMINARY -- eval-only, existing checkpoints", "deltas": recs}
    (OUT / "paired_bootstrap_ci.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    md = ["# P1 paired stratified bootstrap 95% CI (PRELIMINARY -- eval-only)\n",
          f"N={len(y)} target clips, B={B} resamples, stratified by class, paired, "
          f"rng_seed={SEED}. Macro-F1 deltas in percentage points.\n",
          "| Delta | Observed (pp) | 95% CI (pp) | P(delta>0) |",
          "|---|---:|---|---:|"]
    for r in recs:
        md.append(f"| {r['delta']} | {r['observed_pp']:+.2f} | "
                  f"[{r['ci95_pp'][0]:+.2f}, {r['ci95_pp'][1]:+.2f}] | {r['frac_gt0']:.3f} |")
    (OUT / "paired_bootstrap_ci.md").write_text("\n".join(md), encoding="utf-8")
    print("\n[done] wrote", OUT / "paired_bootstrap_ci.md")


if __name__ == "__main__":
    main()
