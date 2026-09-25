# Public baseline -- auto-generated results (fold into PUBLIC_BASELINE_RESULTS.md)

Target = full 418-image 77 GHz; source-val n=162. Final-EMA (ep100), 3 seeds (42/1234/31415). One unified metric path.

## Main table -- SINGLE-model, head-to-head (#1-10)

| # | Method | 77GHz macro-F1 | 77GHz acc | src-val F1 | gen. gap | params (M) | latency ms | boot95 F1 |
|---|---|---|---|---|---|---|---|---|
| 1 | VGG16-BN | 0.117 ± 0.009 | 0.212 ± 0.026 | 0.983 ± 0.006 | +0.867 | 136.38 | - | [0.110, 0.145] |
| 2 | MobileNetV3-Large | 0.146 ± 0.016 | 0.260 ± 0.015 | 0.973 ± 0.003 | +0.828 | 4.54 | - | [0.113, 0.157] |
| 3 | EfficientNet-B0 | 0.089 ± 0.028 | 0.174 ± 0.023 | 0.913 ± 0.032 | +0.824 | 4.34 | - | [0.105, 0.146] |
| 4 | ConvNeXt-Tiny | 0.186 ± 0.024 | 0.282 ± 0.035 | 0.981 ± 0.005 | +0.796 | 28.22 | - | [0.158, 0.217] |
| 5 | Swin-Tiny | 0.071 ± 0.029 | 0.166 ± 0.020 | 0.980 ± 0.003 | +0.908 | 27.92 | - | [0.031, 0.053] |
| 6 | ConvNeXtV2-Tiny | 0.150 ± 0.016 | 0.210 ± 0.015 | 0.973 ± 0.008 | +0.823 | 28.27 | - | [0.119, 0.177] |
| 7 | RadMamba | 0.204 ± 0.030 | 0.245 ± 0.044 | 0.925 ± 0.013 | +0.721 | 0.09 | - | [0.207, 0.275] |
| 8 | SelaFD-ViT-B/16 | 0.107 ± 0.034 | 0.191 ± 0.046 | 0.912 ± 0.019 | +0.805 | 14.48 | - | [0.053, 0.102] |
| 9 | DINOv3 ViT-L/16 | 0.302 ± 0.031 | 0.413 ± 0.025 | 0.970 ± 0.005 | +0.668 | 1.85 | - | [0.298, 0.356] |
| 10 | proposed | 0.832 ± 0.034 | 0.836 ± 0.032 | 0.967 ± 0.006 | +0.135 | 1.85 | - | [0.751, 0.826] |

## Deployment row -- SEPARATE, NOT head-to-head (#11)

| # | Method | 77GHz macro-F1 | 77GHz acc | boot95 F1 |
|---|---|---|---|---|
| 11 | Proposed + 3-seed gap-aware ensemble (deployment) | 0.856 | 0.859 | [0.821, 0.889] |  (cited 0.857/0.859)

## Per-class macro-F1 (3-seed mean, full-418)

| Method | Away | Bend | Kneel | Pick | SStep | Sit | Towards | worst |
|---|---|---|---|---|---|---|---|---|
| VGG16-BN | 0.31 | 0.00 | 0.02 | 0.00 | 0.01 | 0.00 | 0.48 | 0.00 |
| MobileNetV3-Large | 0.44 | 0.00 | 0.06 | 0.00 | 0.07 | 0.00 | 0.45 | 0.00 |
| EfficientNet-B0 | 0.28 | 0.00 | 0.00 | 0.02 | 0.06 | 0.00 | 0.26 | 0.00 |
| ConvNeXt-Tiny | 0.42 | 0.16 | 0.05 | 0.06 | 0.03 | 0.00 | 0.57 | 0.00 |
| Swin-Tiny | 0.26 | 0.00 | 0.00 | 0.00 | 0.04 | 0.00 | 0.20 | 0.00 |
| ConvNeXtV2-Tiny | 0.32 | 0.03 | 0.16 | 0.01 | 0.02 | 0.00 | 0.50 | 0.00 |
| RadMamba | 0.20 | 0.27 | 0.20 | 0.41 | 0.00 | 0.00 | 0.35 | 0.00 |
| SelaFD-ViT-B/16 | 0.39 | 0.02 | 0.03 | 0.00 | 0.08 | 0.00 | 0.23 | 0.00 |
| DINOv3 ViT-L/16 | 0.70 | 0.50 | 0.27 | 0.01 | 0.08 | 0.00 | 0.56 | 0.00 |
| proposed | 0.96 | 0.79 | 0.82 | 0.82 | 0.88 | 0.72 | 0.84 | 0.72 |

## Harness cross-check
proposed-family 278-subset vs history.json max|diff| = **0.0000** (want ~0.0000)
