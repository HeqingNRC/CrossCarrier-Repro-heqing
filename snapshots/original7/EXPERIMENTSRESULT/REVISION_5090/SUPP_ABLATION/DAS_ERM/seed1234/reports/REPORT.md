# V20-V15V13-KnownPeople-UnknownFreq

- Generated: 2026-09-23 21:40:06
- Backbone: vit_large_patch16_dinov3.lvd1689m
- Adapter mode: lora
- Checkpoint: vit_large_patch16_dinov3_lvd1689m_best_sourcequalified77_ema.pt (sourcequalified)
- Selection: source-qualified diagnostic selected (source-val ema acc >= 0.90, then best 77GHz ema macro-F1)

## 77GHz final test

- acc: 0.1619
- macro-F1: 0.0740

### per-class

- Away: 0.0250
- Bend: 1.0000
- Kneel: 0.0000
- Pick: 0.0000
- SStep: 0.0000
- Sit: 0.0000
- Towards: 0.1250

## val (10+24GHz source validation)

- acc: 0.9506
- macro-F1: 0.9509

## Frequency scan on 10GHz val

| f_virt (GHz) | acc | macro-F1 |
|---|---|---|
| 15.0 | 0.7821 | 0.7715 |
| 30.0 | 0.4103 | 0.3395 |
| 50.0 | 0.3077 | 0.2035 |
| 77.0 | 0.3205 | 0.1927 |
| 99.0 | 0.3333 | 0.2048 |
| 120.0 | 0.3333 | 0.2041 |
| 140.0 | 0.3205 | 0.1904 |

## Frequency scan on 24GHz val

| f_virt (GHz) | acc | macro-F1 |
|---|---|---|
| 15.0 | 0.6429 | 0.6286 |
| 30.0 | 0.9167 | 0.9152 |
| 50.0 | 0.5357 | 0.4644 |
| 77.0 | 0.4286 | 0.2863 |
| 99.0 | 0.4286 | 0.2952 |
| 120.0 | 0.4167 | 0.2850 |
| 140.0 | 0.4167 | 0.2992 |
