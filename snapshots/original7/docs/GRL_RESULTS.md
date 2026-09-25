# V13-as-GRL adversarial carrier head — results (2026-06-08)

Replace V13's weak covariance decorrelation with a Gradient-Reversal-Layer (GRL)
adversarial carrier head on `z_cls`. Protocol = FINAL-EMA honest rule
(`pool_ep100_ema.pt`), full-418 77GHz, seeds 42/1234/31415. All runs share the
same FAST_GPU path as the existing A_REF/A_V13/A_V20 baselines (internally
consistent; harness cross-check vs history.json = **max|diff| 0.0000**).

## 1. Implementation (env-gated, default OFF; *.bak_pre_grl backups kept)
- `baseline_v20/config.py`: `V13_GRL_WEIGHT` (0=off), `V13_GRL_HIDDEN` (128),
  `V13_GRL_TARGET` ∈ {shown, base}.
- `baseline_v20/v9_2_1lib.py`: `CarrierAdversary` = Linear(512→128)→ReLU→Dropout(0.2)→Linear(128→1).
- `EXPERIMENTSRESULT/v15_v18_common_train.py`: when `V13_GRL_WEIGHT>0`, build the
  adversary, add to `trainable`. Per step (inside autocast):
  - `lambda = 2/(1+exp(-10·p))-1`, `p = global_step/total_steps` (DANN ramp)
  - target: shown = `log(f_src[d_src]) + r_src` (effective DAS'd carrier);
            base  = `log(f_src[d_src])` (original band)
  - `loss_grl = smooth_l1(adversary(grad_reverse(z_cls, lambda)), target)`
  - `loss_source += V13_GRL_WEIGHT · loss_grl`
- Phase-0 verified: weight=0 reproduces A_V13 (grl=0.0000, ce/v13/acc identical);
  weight>0 fires (adversary in optimizer, grl loss + lambda ramp non-zero).

## 2. Operating configuration
- **GRL weight = 0.3**, shown-carrier target (= log f_src + r_src), adversary on z_cls.
- The GRL erases the *continuous Doppler-scale* (shown-carrier) direction; the base
  10/24 band is baked into the frozen encoder and resists erasure, so the z_cls→carrier
  linear probe stays at ~1.0 regardless of GRL strength (see §3). The gain comes from
  the continuous-scale invariance, not from base-band scrubbing.

## 3. Mechanistic result — the probe does NOT move (negative)
- z_cls→carrier probe, chosen config A_V13_GRL, 3-seed: 0.999 / 1.000 / 0.999
  (**mean 0.999**) — identical to the 0.999 BEFORE. z_freq→carrier stays ~1.000.
- Root cause: with a FROZEN DINOv3 ViT-L + LoRA r2 + small neck, the 10-vs-24
  carrier is a linearly-dominant, baked-in direction. GRL grad reaches the encoder
  (grl loss rises far above the mean-prediction floor: shown w=3.0 grl 0.18→1.32),
  i.e. it DOES erase the *continuous shown carrier*; but a clean-image linear probe
  still separates the *base* band at ~1.0. Pushing harder (base w=3.0) collapses
  class info (0.612) before the probe moves. The task's "probe→chance" mechanism is
  NOT achievable via GRL-on-z_cls in this architecture.

## 4. Performance (final-EMA honest rule, full-418, 3 seeds 42/1234/31415)
| Variant | Accuracy | Macro-F1 | per-seed F1 |
|---|---|---|---|
| E1_noDAS (pure base) | 0.413±0.025 | 0.302±0.031 | .329/.317/.259 |
| A_REF (DAS only) | 0.781±0.070 | 0.767±0.076 | .671/.772/.857 |
| A_V15 (DAS+falsify) | 0.805±0.019 | 0.799±0.020 | .787/.783/.828 |
| A_V13 (DAS+old decorr) | 0.807±0.058 | 0.803±0.058 | .728/.809/.870 |
| A_V20 (full method) | 0.825±0.009 | 0.818±0.009 | .807/.821/.827 |
| **A_V13_GRL (DAS+GRL shown w0.3)** | **0.836±0.032** | **0.832±0.034** | .791/.832/.874 |

A_V13_GRL macro-F1 mean (0.832) is the highest of all variants — above the full
A_V20 method (0.818), though A_V20 has lower variance (±0.009 vs ±0.034).

## 5. Paired analysis vs A_REF (per seed) — IS THE CONTRIBUTION REAL?
| seed | A_REF | A_V13 | A_V13_GRL | Δ_old (V13−REF) | Δ_new (GRL−REF) |
|---|---|---|---|---|---|
| 42 | 0.6707 | 0.7284 | 0.7910 | +0.0577 | +0.1203 |
| 1234 | 0.7720 | 0.8092 | 0.8322 | +0.0373 | +0.0602 |
| 31415 | 0.8573 | 0.8701 | 0.8735 | +0.0128 | +0.0162 |
| **mean** | 0.7667 | 0.8026 | 0.8322 | **+0.0359** | **+0.0655** |

- Δ_new = **+0.0655, 3/3 seeds positive** (no sign flips) and **larger than Δ_old on
  every seed** (.120>.058, .060>.037, .016>.013).
- GRL also **collapses cross-seed variance**: std 0.076 (A_REF) / 0.058 (A_V13) →
  **0.034** (A_V13_GRL). The residual-split contribution becomes real & robust.

## 6. Per-class 77GHz F1 (3-seed mean): A_V13_GRL vs A_V13
| class | A_V13 | A_V13_GRL | Δ |
|---|---|---|---|
| Away | 0.898 | 0.959 | +0.062 |
| Bend | 0.768 | 0.786 | +0.018 |
| Kneel | 0.812 | 0.819 | +0.007 |
| Pick | 0.790 | 0.821 | +0.031 |
| SStep | 0.861 | 0.879 | +0.019 |
| Sit | 0.656 | 0.722 | +0.066 |
| Towards | 0.835 | 0.840 | +0.005 |

All 7 classes improve; the flagged weak classes Sit (+0.066) and Bend (+0.018) both
rise. No regressions. Class-agnostic (Away/Towards not special-cased).

## 7. Verdict
- **Mechanism (task's primary gate): FALSIFIED.** GRL on z_cls does not make z_cls
  carrier-uninformative under the linear probe — the frozen ViT-L bakes the base
  carrier in; the adversary erases the continuous shown carrier but not base-band
  linear separability, and over-pressuring collapses class accuracy first.
- **Performance / paired contribution: REAL & non-noise.** Replacing the weak decorr
  V13 with the GRL (shown-carrier, w=0.3) yields Δ=+0.0655 over the DAS base, 3/3
  seeds positive, larger than old V13 every seed, halves cross-seed variance, lifts
  every class. The mechanism that helps is **continuous Doppler-scale (shown-carrier)
  invariance**, NOT base-carrier scrubbing — a different mechanism than the probe
  measures. So the GRL makes residual-split a real contributor on *performance*, but
  the carrier-probe story does not hold and should not be claimed.
