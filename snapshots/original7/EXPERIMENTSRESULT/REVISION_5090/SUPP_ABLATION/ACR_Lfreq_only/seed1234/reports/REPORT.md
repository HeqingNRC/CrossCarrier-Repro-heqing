# V20-V15V13-KnownPeople-UnknownFreq

- Generated: 2026-09-23 18:01:29
- Backbone: vit_large_patch16_dinov3.lvd1689m
- Adapter mode: lora
- Checkpoint: vit_large_patch16_dinov3_lvd1689m_best_sourcequalified77_ema.pt (sourcequalified)
- Selection: source-qualified diagnostic selected (source-val ema acc >= 0.90, then best 77GHz ema macro-F1)

## 77GHz final test

- acc: 0.8813
- macro-F1: 0.8781

### per-class

- Away: 1.0000
- Bend: 0.9231
- Kneel: 0.8571
- Pick: 0.7500
- SStep: 1.0000
- Sit: 0.6486
- Towards: 0.9750

## val (10+24GHz source validation)

- acc: 0.9691
- macro-F1: 0.9694

## Frequency scan on 10GHz val

| f_virt (GHz) | acc | macro-F1 |
|---|---|---|
| 15.0 | 0.9744 | 0.9756 |
| 30.0 | 0.9744 | 0.9760 |
| 50.0 | 0.9872 | 0.9881 |
| 77.0 | 0.9615 | 0.9642 |
| 99.0 | 0.9615 | 0.9642 |
| 120.0 | 0.9359 | 0.9357 |
| 140.0 | 0.9359 | 0.9361 |

## Frequency scan on 24GHz val

| f_virt (GHz) | acc | macro-F1 |
|---|---|---|
| 15.0 | 0.9762 | 0.9760 |
| 30.0 | 0.9762 | 0.9751 |
| 50.0 | 0.9881 | 0.9875 |
| 77.0 | 0.9643 | 0.9637 |
| 99.0 | 0.9762 | 0.9761 |
| 120.0 | 0.9405 | 0.9406 |
| 140.0 | 0.9167 | 0.9165 |
