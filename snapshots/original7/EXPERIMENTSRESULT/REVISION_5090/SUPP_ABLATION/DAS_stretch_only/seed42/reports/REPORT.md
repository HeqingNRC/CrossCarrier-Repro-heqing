# V20-V15V13-KnownPeople-UnknownFreq

- Generated: 2026-09-23 22:43:41
- Backbone: vit_large_patch16_dinov3.lvd1689m
- Adapter mode: lora
- Checkpoint: vit_large_patch16_dinov3_lvd1689m_best_sourcequalified77_ema.pt (sourcequalified)
- Selection: source-qualified diagnostic selected (source-val ema acc >= 0.90, then best 77GHz ema macro-F1)

## 77GHz final test

- acc: 0.6835
- macro-F1: 0.6626

### per-class

- Away: 1.0000
- Bend: 0.6667
- Kneel: 0.7857
- Pick: 0.7250
- SStep: 0.7250
- Sit: 0.1081
- Towards: 0.7250

## val (10+24GHz source validation)

- acc: 0.9630
- macro-F1: 0.9640

## Frequency scan on 10GHz val

| f_virt (GHz) | acc | macro-F1 |
|---|---|---|
| 15.0 | 0.9744 | 0.9760 |
| 30.0 | 0.9615 | 0.9601 |
| 50.0 | 0.9872 | 0.9881 |
| 77.0 | 0.9744 | 0.9762 |
| 99.0 | 0.9487 | 0.9520 |
| 120.0 | 0.9231 | 0.9270 |
| 140.0 | 0.9103 | 0.9070 |

## Frequency scan on 24GHz val

| f_virt (GHz) | acc | macro-F1 |
|---|---|---|
| 15.0 | 0.9762 | 0.9760 |
| 30.0 | 0.9762 | 0.9760 |
| 50.0 | 0.9881 | 0.9875 |
| 77.0 | 0.9762 | 0.9751 |
| 99.0 | 0.9762 | 0.9751 |
| 120.0 | 0.9643 | 0.9627 |
| 140.0 | 0.9286 | 0.9261 |
