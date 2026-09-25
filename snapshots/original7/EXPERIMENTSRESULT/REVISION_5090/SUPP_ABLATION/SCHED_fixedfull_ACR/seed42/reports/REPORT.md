# V20-V15V13-KnownPeople-UnknownFreq

- Generated: 2026-09-23 23:46:58
- Backbone: vit_large_patch16_dinov3.lvd1689m
- Adapter mode: lora
- Checkpoint: vit_large_patch16_dinov3_lvd1689m_best_sourcequalified77_ema.pt (sourcequalified)
- Selection: source-qualified diagnostic selected (source-val ema acc >= 0.90, then best 77GHz ema macro-F1)

## 77GHz final test

- acc: 0.6223
- macro-F1: 0.6060

### per-class

- Away: 0.0500
- Bend: 0.7179
- Kneel: 0.9762
- Pick: 0.9500
- SStep: 1.0000
- Sit: 0.3784
- Towards: 0.2500

## val (10+24GHz source validation)

- acc: 0.9630
- macro-F1: 0.9628

## Frequency scan on 10GHz val

| f_virt (GHz) | acc | macro-F1 |
|---|---|---|
| 15.0 | 0.9744 | 0.9760 |
| 30.0 | 0.9872 | 0.9881 |
| 50.0 | 0.9872 | 0.9881 |
| 77.0 | 0.9744 | 0.9760 |
| 99.0 | 0.9615 | 0.9637 |
| 120.0 | 0.9487 | 0.9519 |
| 140.0 | 0.9487 | 0.9519 |

## Frequency scan on 24GHz val

| f_virt (GHz) | acc | macro-F1 |
|---|---|---|
| 15.0 | 0.9762 | 0.9755 |
| 30.0 | 1.0000 | 1.0000 |
| 50.0 | 1.0000 | 1.0000 |
| 77.0 | 1.0000 | 1.0000 |
| 99.0 | 1.0000 | 1.0000 |
| 120.0 | 0.9881 | 0.9875 |
| 140.0 | 1.0000 | 1.0000 |
