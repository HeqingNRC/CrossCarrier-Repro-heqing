# V20-V15V13-KnownPeople-UnknownFreq

- Generated: 2026-09-23 18:09:37
- Backbone: vit_large_patch16_dinov3.lvd1689m
- Adapter mode: lora
- Checkpoint: vit_large_patch16_dinov3_lvd1689m_best_sourcequalified77_ema.pt (sourcequalified)
- Selection: source-qualified diagnostic selected (source-val ema acc >= 0.90, then best 77GHz ema macro-F1)

## 77GHz final test

- acc: 0.8813
- macro-F1: 0.8792

### per-class

- Away: 0.9750
- Bend: 0.7436
- Kneel: 0.7857
- Pick: 0.9500
- SStep: 1.0000
- Sit: 0.7838
- Towards: 0.9250

## val (10+24GHz source validation)

- acc: 0.9630
- macro-F1: 0.9634

## Frequency scan on 10GHz val

| f_virt (GHz) | acc | macro-F1 |
|---|---|---|
| 15.0 | 0.9744 | 0.9760 |
| 30.0 | 0.9872 | 0.9881 |
| 50.0 | 0.9872 | 0.9881 |
| 77.0 | 0.9744 | 0.9762 |
| 99.0 | 0.9744 | 0.9762 |
| 120.0 | 0.9359 | 0.9392 |
| 140.0 | 0.9103 | 0.9144 |

## Frequency scan on 24GHz val

| f_virt (GHz) | acc | macro-F1 |
|---|---|---|
| 15.0 | 0.9762 | 0.9751 |
| 30.0 | 0.9524 | 0.9517 |
| 50.0 | 0.9643 | 0.9631 |
| 77.0 | 0.9643 | 0.9627 |
| 99.0 | 0.9643 | 0.9631 |
| 120.0 | 0.9643 | 0.9646 |
| 140.0 | 0.9405 | 0.9403 |
