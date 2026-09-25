# V20-V15V13-KnownPeople-UnknownFreq

- Generated: 2026-09-24 00:51:19
- Backbone: vit_large_patch16_dinov3.lvd1689m
- Adapter mode: lora
- Checkpoint: vit_large_patch16_dinov3_lvd1689m_best_sourcequalified77_ema.pt (sourcequalified)
- Selection: source-qualified diagnostic selected (source-val ema acc >= 0.90, then best 77GHz ema macro-F1)

## 77GHz final test

- acc: 0.8597
- macro-F1: 0.8412

### per-class

- Away: 1.0000
- Bend: 0.7949
- Kneel: 1.0000
- Pick: 0.9000
- SStep: 1.0000
- Sit: 0.3514
- Towards: 0.9250

## val (10+24GHz source validation)

- acc: 0.9815
- macro-F1: 0.9817

## Frequency scan on 10GHz val

| f_virt (GHz) | acc | macro-F1 |
|---|---|---|
| 15.0 | 0.9744 | 0.9760 |
| 30.0 | 0.9615 | 0.9637 |
| 50.0 | 0.9359 | 0.9347 |
| 77.0 | 0.9103 | 0.9110 |
| 99.0 | 0.8462 | 0.8470 |
| 120.0 | 0.8333 | 0.8294 |
| 140.0 | 0.7436 | 0.7328 |

## Frequency scan on 24GHz val

| f_virt (GHz) | acc | macro-F1 |
|---|---|---|
| 15.0 | 0.9881 | 0.9876 |
| 30.0 | 0.9881 | 0.9875 |
| 50.0 | 0.9643 | 0.9646 |
| 77.0 | 0.8810 | 0.8773 |
| 99.0 | 0.8214 | 0.7955 |
| 120.0 | 0.7500 | 0.7033 |
| 140.0 | 0.7619 | 0.7173 |
