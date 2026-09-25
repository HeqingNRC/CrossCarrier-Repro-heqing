# Component ablation -- final-EMA honest rule, full-418 77GHz, 3 seeds (42/1234/31415)

Selection = EMA weights at the last epoch (ep100); no target/source-val peeking.

| Variant | Accuracy | Macro-F1 | d macro-F1 (pp) |
|---|---|---|---:|
| ACR L_freq only | 0.822 ± 0.034 | 0.818 ± 0.035 | +0.0 |
| ACR GRL only | 0.801 ± 0.066 | 0.789 ± 0.080 | -2.9 |
| ACR discrete adversary | 0.803 ± 0.030 | 0.794 ± 0.037 | -2.4 |
| DAS ERM (no aug) | 0.193 ± 0.051 | 0.087 ± 0.034 | -73.1 |
| DAS stretch only | 0.698 ± 0.018 | 0.677 ± 0.017 | -14.1 |
| DAS fixed-full + ACR | 0.620 ± 0.022 | 0.584 ± 0.019 | -23.4 |
| DAS fixed-narrow + ACR | 0.844 ± 0.020 | 0.835 ± 0.027 | +1.6 |

Per-seed Macro-F1 (full-418):

| Variant | seed42 | seed1234 | seed31415 |
|---|---:|---:|---:|
| ACR L_freq only | 0.7701 | 0.8501 | 0.8343 |
| ACR GRL only | 0.6763 | 0.8522 | 0.8393 |
| ACR discrete adversary | 0.7417 | 0.8144 | 0.8266 |
| DAS ERM (no aug) | 0.1285 | 0.0871 | 0.0448 |
| DAS stretch only | 0.6549 | 0.6973 | 0.6782 |
| DAS fixed-full + ACR | 0.5781 | 0.6091 | 0.5644 |
| DAS fixed-narrow + ACR | 0.8112 | 0.8725 | 0.8198 |

## Component contributions (full-418 macro-F1, mean)
- Doppler-stretch alone over ERM: DAS_stretch_only - DAS_ERM = **+59.0 pp**

## Eval-harness cross-check (278-subset final-EMA vs run history last-epoch)
max |diff| vs history.json = 0.0000 (should be ~0 -> harness validated)