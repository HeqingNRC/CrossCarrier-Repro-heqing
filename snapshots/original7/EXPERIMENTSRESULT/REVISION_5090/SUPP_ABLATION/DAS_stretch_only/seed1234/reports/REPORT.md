# V20-V15V13-KnownPeople-UnknownFreq

- Generated: 2026-09-23 22:44:41
- Backbone: vit_large_patch16_dinov3.lvd1689m
- Adapter mode: lora
- Checkpoint: vit_large_patch16_dinov3_lvd1689m_best_sourcequalified77_ema.pt (sourcequalified)
- Selection: source-qualified diagnostic selected (source-val ema acc >= 0.90, then best 77GHz ema macro-F1)

## 77GHz final test

- acc: 0.7338
- macro-F1: 0.7364

### per-class

- Away: 0.9750
- Bend: 0.9487
- Kneel: 0.6429
- Pick: 0.3500
- SStep: 1.0000
- Sit: 0.5405
- Towards: 0.6750

## val (10+24GHz source validation)

- acc: 0.9753
- macro-F1: 0.9749

## Frequency scan on 10GHz val

| f_virt (GHz) | acc | macro-F1 |
|---|---|---|
| 15.0 | 0.9487 | 0.9466 |
| 30.0 | 0.9615 | 0.9593 |
| 50.0 | 0.9872 | 0.9881 |
| 77.0 | 0.9872 | 0.9881 |
| 99.0 | 0.9615 | 0.9642 |
| 120.0 | 0.9231 | 0.9260 |
| 140.0 | 0.8974 | 0.9000 |

## Frequency scan on 24GHz val

| f_virt (GHz) | acc | macro-F1 |
|---|---|---|
| 15.0 | 1.0000 | 1.0000 |
| 30.0 | 0.9881 | 0.9875 |
| 50.0 | 0.9762 | 0.9760 |
| 77.0 | 0.9762 | 0.9760 |
| 99.0 | 0.9643 | 0.9637 |
| 120.0 | 0.9762 | 0.9760 |
| 140.0 | 0.9643 | 0.9641 |
