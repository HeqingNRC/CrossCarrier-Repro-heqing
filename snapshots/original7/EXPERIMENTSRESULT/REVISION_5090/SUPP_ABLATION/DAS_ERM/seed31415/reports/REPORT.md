# V20-V15V13-KnownPeople-UnknownFreq

- Generated: 2026-09-23 21:40:05
- Backbone: vit_large_patch16_dinov3.lvd1689m
- Adapter mode: lora
- Checkpoint: vit_large_patch16_dinov3_lvd1689m_best_sourcequalified77_ema.pt (sourcequalified)
- Selection: source-qualified diagnostic selected (source-val ema acc >= 0.90, then best 77GHz ema macro-F1)

## 77GHz final test

- acc: 0.2446
- macro-F1: 0.1097

### per-class

- Away: 0.0000
- Bend: 0.7436
- Kneel: 0.0000
- Pick: 0.0000
- SStep: 0.0000
- Sit: 0.0000
- Towards: 0.9750

## val (10+24GHz source validation)

- acc: 0.9259
- macro-F1: 0.9270

## Frequency scan on 10GHz val

| f_virt (GHz) | acc | macro-F1 |
|---|---|---|
| 15.0 | 0.6538 | 0.6045 |
| 30.0 | 0.3205 | 0.2227 |
| 50.0 | 0.3333 | 0.2365 |
| 77.0 | 0.3974 | 0.2582 |
| 99.0 | 0.3462 | 0.2292 |
| 120.0 | 0.2821 | 0.1719 |
| 140.0 | 0.2564 | 0.1438 |

## Frequency scan on 24GHz val

| f_virt (GHz) | acc | macro-F1 |
|---|---|---|
| 15.0 | 0.6190 | 0.5756 |
| 30.0 | 0.8929 | 0.8942 |
| 50.0 | 0.3333 | 0.2381 |
| 77.0 | 0.2976 | 0.2067 |
| 99.0 | 0.2857 | 0.1942 |
| 120.0 | 0.2738 | 0.1775 |
| 140.0 | 0.2500 | 0.1616 |
