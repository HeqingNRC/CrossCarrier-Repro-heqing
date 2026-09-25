# GRL_w0.3

- Generated: 2026-06-08 09:49:46
- Backbone: vit_large_patch16_dinov3.lvd1689m
- Adapter mode: lora
- Checkpoint: vit_large_patch16_dinov3_lvd1689m_best_sourcequalified77_ema.pt (sourcequalified)
- Selection: source-qualified diagnostic selected (source-val ema acc >= 0.00, then best 77GHz ema macro-F1)

## 77GHz final test

- acc: 0.8201
- macro-F1: 0.8122

### per-class

- Away: 0.9750
- Bend: 0.7949
- Kneel: 0.8571
- Pick: 0.8750
- SStep: 1.0000
- Sit: 0.4324
- Towards: 0.7750

## val (10+24GHz source validation)

- acc: 0.9753
- macro-F1: 0.9749

## Frequency scan on 10GHz val

| f_virt (GHz) | acc | macro-F1 |
|---|---|---|
| 15.0 | 0.9744 | 0.9760 |
| 30.0 | 0.9615 | 0.9593 |
| 50.0 | 0.9744 | 0.9756 |
| 77.0 | 0.9615 | 0.9642 |
| 99.0 | 0.9615 | 0.9642 |
| 120.0 | 0.9615 | 0.9642 |
| 140.0 | 0.8974 | 0.9018 |

## Frequency scan on 24GHz val

| f_virt (GHz) | acc | macro-F1 |
|---|---|---|
| 15.0 | 0.9881 | 0.9876 |
| 30.0 | 0.9881 | 0.9885 |
| 50.0 | 0.9881 | 0.9885 |
| 77.0 | 0.9762 | 0.9760 |
| 99.0 | 0.9643 | 0.9644 |
| 120.0 | 0.9643 | 0.9627 |
| 140.0 | 0.9524 | 0.9501 |
