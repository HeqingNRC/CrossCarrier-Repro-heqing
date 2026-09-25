# V20-V15V13-KnownPeople-UnknownFreq

- Generated: 2026-09-23 19:19:38
- Backbone: vit_large_patch16_dinov3.lvd1689m
- Adapter mode: lora
- Checkpoint: vit_large_patch16_dinov3_lvd1689m_best_sourcequalified77_ema.pt (sourcequalified)
- Selection: source-qualified diagnostic selected (source-val ema acc >= 0.90, then best 77GHz ema macro-F1)

## 77GHz final test

- acc: 0.8669
- macro-F1: 0.8658

### per-class

- Away: 1.0000
- Bend: 0.8205
- Kneel: 0.8810
- Pick: 0.8000
- SStep: 1.0000
- Sit: 0.6757
- Towards: 0.8750

## val (10+24GHz source validation)

- acc: 0.9691
- macro-F1: 0.9694

## Frequency scan on 10GHz val

| f_virt (GHz) | acc | macro-F1 |
|---|---|---|
| 15.0 | 0.9744 | 0.9760 |
| 30.0 | 0.9744 | 0.9760 |
| 50.0 | 0.9744 | 0.9760 |
| 77.0 | 0.9487 | 0.9478 |
| 99.0 | 0.9487 | 0.9520 |
| 120.0 | 0.9359 | 0.9361 |
| 140.0 | 0.8846 | 0.8907 |

## Frequency scan on 24GHz val

| f_virt (GHz) | acc | macro-F1 |
|---|---|---|
| 15.0 | 0.9881 | 0.9876 |
| 30.0 | 0.9762 | 0.9751 |
| 50.0 | 0.9643 | 0.9627 |
| 77.0 | 0.9762 | 0.9760 |
| 99.0 | 0.9643 | 0.9646 |
| 120.0 | 0.9524 | 0.9513 |
| 140.0 | 0.9643 | 0.9637 |
