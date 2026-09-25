# V20-V15V13-KnownPeople-UnknownFreq

- Generated: 2026-09-23 22:45:54
- Backbone: vit_large_patch16_dinov3.lvd1689m
- Adapter mode: lora
- Checkpoint: vit_large_patch16_dinov3_lvd1689m_best_sourcequalified77_ema.pt (sourcequalified)
- Selection: source-qualified diagnostic selected (source-val ema acc >= 0.90, then best 77GHz ema macro-F1)

## 77GHz final test

- acc: 0.7302
- macro-F1: 0.6971

### per-class

- Away: 0.9500
- Bend: 0.7179
- Kneel: 0.8571
- Pick: 0.7750
- SStep: 0.9250
- Sit: 0.1081
- Towards: 0.7250

## val (10+24GHz source validation)

- acc: 0.9444
- macro-F1: 0.9450

## Frequency scan on 10GHz val

| f_virt (GHz) | acc | macro-F1 |
|---|---|---|
| 15.0 | 0.9615 | 0.9637 |
| 30.0 | 0.9872 | 0.9881 |
| 50.0 | 0.9615 | 0.9637 |
| 77.0 | 0.9359 | 0.9385 |
| 99.0 | 0.9487 | 0.9519 |
| 120.0 | 0.9487 | 0.9519 |
| 140.0 | 0.9359 | 0.9388 |

## Frequency scan on 24GHz val

| f_virt (GHz) | acc | macro-F1 |
|---|---|---|
| 15.0 | 0.9762 | 0.9751 |
| 30.0 | 0.9524 | 0.9513 |
| 50.0 | 0.9405 | 0.9398 |
| 77.0 | 0.9524 | 0.9513 |
| 99.0 | 0.9643 | 0.9627 |
| 120.0 | 0.9524 | 0.9513 |
| 140.0 | 0.9286 | 0.9274 |
