# V20-V15V13-KnownPeople-UnknownFreq

- Generated: 2026-09-23 20:31:25
- Backbone: vit_large_patch16_dinov3.lvd1689m
- Adapter mode: lora
- Checkpoint: vit_large_patch16_dinov3_lvd1689m_best_sourcequalified77_ema.pt (sourcequalified)
- Selection: source-qualified diagnostic selected (source-val ema acc >= 0.90, then best 77GHz ema macro-F1)

## 77GHz final test

- acc: 0.8489
- macro-F1: 0.8458

### per-class

- Away: 0.9750
- Bend: 0.7949
- Kneel: 0.8095
- Pick: 0.9250
- SStep: 1.0000
- Sit: 0.7568
- Towards: 0.6750

## val (10+24GHz source validation)

- acc: 0.9815
- macro-F1: 0.9816

## Frequency scan on 10GHz val

| f_virt (GHz) | acc | macro-F1 |
|---|---|---|
| 15.0 | 0.9744 | 0.9760 |
| 30.0 | 0.9615 | 0.9601 |
| 50.0 | 0.9744 | 0.9760 |
| 77.0 | 0.9615 | 0.9637 |
| 99.0 | 0.9487 | 0.9518 |
| 120.0 | 0.9359 | 0.9395 |
| 140.0 | 0.9231 | 0.9251 |

## Frequency scan on 24GHz val

| f_virt (GHz) | acc | macro-F1 |
|---|---|---|
| 15.0 | 0.9881 | 0.9885 |
| 30.0 | 0.9881 | 0.9876 |
| 50.0 | 0.9881 | 0.9885 |
| 77.0 | 0.9881 | 0.9875 |
| 99.0 | 0.9762 | 0.9760 |
| 120.0 | 0.9881 | 0.9875 |
| 140.0 | 0.9643 | 0.9641 |
