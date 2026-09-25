# V20-V15V13-KnownPeople-UnknownFreq

- Generated: 2026-09-24 00:53:00
- Backbone: vit_large_patch16_dinov3.lvd1689m
- Adapter mode: lora
- Checkpoint: vit_large_patch16_dinov3_lvd1689m_best_sourcequalified77_ema.pt (sourcequalified)
- Selection: source-qualified diagnostic selected (source-val ema acc >= 0.90, then best 77GHz ema macro-F1)

## 77GHz final test

- acc: 0.8453
- macro-F1: 0.8361

### per-class

- Away: 1.0000
- Bend: 0.7692
- Kneel: 0.9286
- Pick: 0.7250
- SStep: 1.0000
- Sit: 0.4865
- Towards: 0.9750

## val (10+24GHz source validation)

- acc: 0.9877
- macro-F1: 0.9878

## Frequency scan on 10GHz val

| f_virt (GHz) | acc | macro-F1 |
|---|---|---|
| 15.0 | 0.9744 | 0.9760 |
| 30.0 | 0.9872 | 0.9881 |
| 50.0 | 0.9487 | 0.9478 |
| 77.0 | 0.9231 | 0.9189 |
| 99.0 | 0.8846 | 0.8867 |
| 120.0 | 0.8462 | 0.8503 |
| 140.0 | 0.8333 | 0.8390 |

## Frequency scan on 24GHz val

| f_virt (GHz) | acc | macro-F1 |
|---|---|---|
| 15.0 | 0.9881 | 0.9876 |
| 30.0 | 0.9881 | 0.9876 |
| 50.0 | 0.9524 | 0.9513 |
| 77.0 | 0.7976 | 0.7727 |
| 99.0 | 0.6429 | 0.5973 |
| 120.0 | 0.5595 | 0.5153 |
| 140.0 | 0.5119 | 0.4618 |
