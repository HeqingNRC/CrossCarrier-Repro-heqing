# A_V13_GRL

- Generated: 2026-06-08 16:38:30
- Backbone: vit_large_patch16_dinov3.lvd1689m
- Adapter mode: lora
- Checkpoint: vit_large_patch16_dinov3_lvd1689m_best_sourcequalified77_ema.pt (sourcequalified)
- Selection: source-qualified diagnostic selected (source-val ema acc >= 0.00, then best 77GHz ema macro-F1)

## 77GHz final test

- acc: 0.8669
- macro-F1: 0.8641

### per-class

- Away: 1.0000
- Bend: 0.7949
- Kneel: 0.8095
- Pick: 0.9250
- SStep: 1.0000
- Sit: 0.7027
- Towards: 0.8250

## val (10+24GHz source validation)

- acc: 0.9630
- macro-F1: 0.9627

## Frequency scan on 10GHz val

| f_virt (GHz) | acc | macro-F1 |
|---|---|---|
| 15.0 | 0.9615 | 0.9637 |
| 30.0 | 0.9744 | 0.9760 |
| 50.0 | 0.9744 | 0.9760 |
| 77.0 | 0.9744 | 0.9760 |
| 99.0 | 0.9615 | 0.9642 |
| 120.0 | 0.9487 | 0.9518 |
| 140.0 | 0.9103 | 0.9145 |

## Frequency scan on 24GHz val

| f_virt (GHz) | acc | macro-F1 |
|---|---|---|
| 15.0 | 0.9881 | 0.9881 |
| 30.0 | 0.9762 | 0.9755 |
| 50.0 | 0.9762 | 0.9760 |
| 77.0 | 0.9762 | 0.9760 |
| 99.0 | 0.9643 | 0.9646 |
| 120.0 | 0.9524 | 0.9522 |
| 140.0 | 0.9524 | 0.9522 |
