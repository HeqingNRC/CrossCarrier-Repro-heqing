# V20-V15V13-KnownPeople-UnknownFreq

- Generated: 2026-09-23 20:31:23
- Backbone: vit_large_patch16_dinov3.lvd1689m
- Adapter mode: lora
- Checkpoint: vit_large_patch16_dinov3_lvd1689m_best_sourcequalified77_ema.pt (sourcequalified)
- Selection: source-qualified diagnostic selected (source-val ema acc >= 0.90, then best 77GHz ema macro-F1)

## 77GHz final test

- acc: 0.8453
- macro-F1: 0.8424

### per-class

- Away: 1.0000
- Bend: 0.7692
- Kneel: 0.8333
- Pick: 0.8000
- SStep: 1.0000
- Sit: 0.6486
- Towards: 0.8500

## val (10+24GHz source validation)

- acc: 0.9691
- macro-F1: 0.9694

## Frequency scan on 10GHz val

| f_virt (GHz) | acc | macro-F1 |
|---|---|---|
| 15.0 | 0.9744 | 0.9760 |
| 30.0 | 0.9615 | 0.9642 |
| 50.0 | 0.9744 | 0.9760 |
| 77.0 | 0.9744 | 0.9760 |
| 99.0 | 0.9615 | 0.9642 |
| 120.0 | 0.9487 | 0.9518 |
| 140.0 | 0.9231 | 0.9235 |

## Frequency scan on 24GHz val

| f_virt (GHz) | acc | macro-F1 |
|---|---|---|
| 15.0 | 0.9881 | 0.9876 |
| 30.0 | 0.9762 | 0.9757 |
| 50.0 | 0.9643 | 0.9637 |
| 77.0 | 0.9762 | 0.9751 |
| 99.0 | 0.9643 | 0.9637 |
| 120.0 | 0.9762 | 0.9751 |
| 140.0 | 0.9643 | 0.9632 |
