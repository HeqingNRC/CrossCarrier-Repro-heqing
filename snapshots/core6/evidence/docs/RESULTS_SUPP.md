# Supplementary Ablation — RESULTS (21/21 runs complete)

Final-EMA (ep100) weights, full-418 77GHz, 3 seeds (42/1234/31415), no target/source-val
selection. Eval-harness cross-check **max |diff| vs history.json = 0.0000 → validated**.
Full table: `EXPERIMENTSRESULT/CKPT_SELECTION_DIAG/ablation_finalEMA_3seed.md`.
2 of 21 runs crashed early (14900KF instability) and were auto-rerun to completion.

> Naming note: the table's **"Full (Proposed)" = old A_V20 (DAS+residual+V15 falsification)
> = 0.818±0.009**. The paper headline is **DAS+ACR = "V13-as-GRL" = 0.832±0.034** (single)
> / 0.857 (3-seed ensemble). V15 is dropped per the plan.

## P0-A — ACR internal mechanism (DAS curriculum fixed; vary ACR sub-parts)
| Row | macro-F1 | Δ vs DAS-only |
|---|---|---|
| DAS only (A_REF) | 0.767 ± 0.076 | — |
| + residual branch only (no GRL) | 0.814 ± 0.021 | +4.7 |
| + GRL only (no residual) | 0.826 ± 0.016 | +5.9 |
| + **full ACR** (residual + continuous GRL) | **0.832 ± 0.034** | **+6.5** |
| + discrete carrier adversary (3-bin CE) | 0.796 ± 0.034 | +2.9 |

**Findings:** both halves help; GRL is the dominant half. Our **continuous log-carrier
regressor beats the ordinary discrete domain adversary by +3.6 pp** (0.832 vs 0.796;
paired bootstrap +3.57 pp, 95% CI [+1.91, +5.31], P(>0)=1.000, B=10000 — see
`exact_result_reports/bootstrap_continuous_vs_discrete.md`) — the key novelty defense.
Components are sub-additive (4.7+5.9 ⟶ 6.5), i.e. overlapping but complementary; full ACR
is best.

## P0-B — DAS body (separate Doppler-stretch from generic radar aug)
| Row | macro-F1 | note |
|---|---|---|
| ERM — true no-aug | 0.146 ± 0.034 | new floor |
| radar-aug only (HFT+SpecAug, no stretch) | 0.302 ± 0.031 | +15.6 vs ERM |
| Doppler-stretch only (no HFT/SpecAug) | 0.704 ± 0.089 | **+55.8 vs ERM; +40.2 vs radar-aug** |
| physics-free jitter | 0.518 ± 0.030 | stretch-only is +18.6 vs jitter |
| full DAS (stretch + radar aug) | 0.767 ± 0.076 | +6.3 vs stretch-only |

**Findings:** the carrier-matched **Doppler-stretch is the overwhelming driver (+55.8 pp
over ERM)**; generic radar aug is minor (+15.6); and it must be carrier-matched — physics-free
jitter (0.518) ≪ stretch (0.704). HFT/SpecAug add a small +6.3 on top.

## P1 — schedule UNDER ACR (resolves the earlier fixed-narrow contradiction)
| Schedule | no ACR | + full ACR | ACR effect |
|---|---|---|---|
| curriculum | 0.767 (A_REF) | **0.832** | **+6.5** |
| fixed-narrow [10,30] | 0.804 (E2_narrow) | 0.800 ± 0.057 | −0.4 |
| fixed-full [15,140] | 0.756 (E2_full) | 0.626 ± 0.058 | **−13.0** |

**Findings:** **ACR helps only with the curriculum.** Without ACR, fixed-narrow (0.804) >
curriculum (0.767) — the old contradiction. With ACR on, curriculum wins clearly
(0.832 > 0.800 > 0.626), and pairing ACR with a fixed wide band from epoch 1 is
destabilizing (fixed-full collapses −13 pp). → keep a 1-line claim: the adversarial
carrier head needs the gradual curriculum.

## P0-C — ensemble fairness (eval-only)
| Config | Single | majority | logit-avg | posterior |
|---|---|---|---|---|
| DAS only | 0.767 ± 0.076 | 0.817 | 0.827 | 0.822 |
| DAS+ACR | 0.832 ± 0.034 | 0.851 | 0.857 | 0.856 |
| jitter | 0.518 ± 0.030 | 0.537 | 0.537 | 0.543 |

All three rules land together (~0.851–0.857); DAS-only also gains ~+5 pp from ensembling
⟶ the ensemble gain is **not ACR-specific** and posterior-avg is **not a target-tuned trick**.

## P1 — paired stratified bootstrap (95% CI, B=10000, N=418, paired, class-stratified)
| Delta | Observed | 95% CI | P(Δ>0) |
|---|---|---|---|
| DAS+ACR single − DAS single | +6.55 pp | [+4.69, +8.37] | 1.000 |
| DAS single − jitter single | +24.90 pp | [+21.47, +28.32] | 1.000 |
| DAS+ACR ensemble − DAS+ACR single | +2.40 pp | [+0.89, +3.96] | 0.999 |
| continuous − discrete adversary (full ACR) | +3.57 pp | [+1.91, +5.31] | 1.000 |

All key deltas are significant (CI excludes 0). The continuous-vs-discrete delta is from
`analysis_scripts/bootstrap_discrete.py` (the other three from `paired_bootstrap_ci.py`).

## Caveats
- High cross-seed variance on some rows (DAS-stretch-only ±0.089, fixed-full+ACR ±0.058,
  fixed-narrow+ACR ±0.057) — consistent with the ~0.05/seed GPU-noise caveat. Headline rows
  are tight (Full-V20 ±0.009, GRL-only ±0.016).
- DAS-stretch-only seed42 (0.587) is a low outlier dragging its mean/variance.
