# Public baseline -- auto-generated results (fold into PUBLIC_BASELINE_RESULTS.md)

Target = full 418-image 77 GHz; source-val n=162. Final-EMA (ep100), 3 seeds (42/1234/31415). One unified metric path.

## Main table -- SINGLE-model, head-to-head (#1-10)

| # | Method | 77GHz macro-F1 | 77GHz acc | src-val F1 | gen. gap | params (M) | latency ms | boot95 F1 |
|---|---|---|---|---|---|---|---|---|
| 10 | proposed | 0.832 ± 0.034 | 0.836 ± 0.032 | 0.967 ± 0.006 | +0.135 | 1.85 | - | [0.751, 0.826] |

## Deployment row -- SEPARATE, NOT head-to-head (#11)

| # | Method | 77GHz macro-F1 | 77GHz acc | boot95 F1 |
|---|---|---|---|---|
| 11 | Proposed + 3-seed gap-aware ensemble (deployment) | 0.856 | 0.859 | [0.821, 0.889] |  (cited 0.857/0.859)

## Per-class macro-F1 (3-seed mean, full-418)

| Method | Away | Bend | Kneel | Pick | SStep | Sit | Towards | worst |
|---|---|---|---|---|---|---|---|---|
| proposed | 0.96 | 0.79 | 0.82 | 0.82 | 0.88 | 0.72 | 0.84 | 0.72 |

## Harness cross-check
proposed-family 278-subset vs history.json max|diff| = **0.0000** (want ~0.0000)
