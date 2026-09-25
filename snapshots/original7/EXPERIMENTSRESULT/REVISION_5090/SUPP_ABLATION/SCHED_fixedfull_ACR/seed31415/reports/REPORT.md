# V20-V15V13-KnownPeople-UnknownFreq

- Generated: 2026-09-23 23:48:30
- Backbone: vit_large_patch16_dinov3.lvd1689m
- Adapter mode: lora
- Checkpoint: vit_large_patch16_dinov3_lvd1689m_best_sourcequalified77_ema.pt (sourcequalified)
- Selection: source-qualified diagnostic selected (source-val ema acc >= 0.90, then best 77GHz ema macro-F1)

## 77GHz final test

- acc: 0.6475
- macro-F1: 0.5894

### per-class

- Away: 0.0000
- Bend: 0.7436
- Kneel: 0.9048
- Pick: 0.7750
- SStep: 0.9750
- Sit: 0.2432
- Towards: 0.8500

## val (10+24GHz source validation)

- acc: 0.9691
- macro-F1: 0.9692

## Frequency scan on 10GHz val

| f_virt (GHz) | acc | macro-F1 |
|---|---|---|
| 15.0 | 0.9744 | 0.9760 |
| 30.0 | 0.9744 | 0.9760 |
| 50.0 | 0.9744 | 0.9760 |
| 77.0 | 0.9744 | 0.9760 |
| 99.0 | 0.9615 | 0.9637 |
| 120.0 | 0.9615 | 0.9626 |
| 140.0 | 0.9615 | 0.9626 |

## Frequency scan on 24GHz val

| f_virt (GHz) | acc | macro-F1 |
|---|---|---|
| 15.0 | 0.9643 | 0.9645 |
| 30.0 | 0.9762 | 0.9755 |
| 50.0 | 0.9762 | 0.9760 |
| 77.0 | 0.9762 | 0.9760 |
| 99.0 | 0.9881 | 0.9875 |
| 120.0 | 0.9643 | 0.9646 |
| 140.0 | 0.9881 | 0.9875 |
