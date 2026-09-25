# V20-V15V13-KnownPeople-UnknownFreq

- Generated: 2026-09-23 21:41:02
- Backbone: vit_large_patch16_dinov3.lvd1689m
- Adapter mode: lora
- Checkpoint: vit_large_patch16_dinov3_lvd1689m_best_sourcequalified77_ema.pt (sourcequalified)
- Selection: source-qualified diagnostic selected (source-val ema acc >= 0.90, then best 77GHz ema macro-F1)

## 77GHz final test

- acc: 0.2374
- macro-F1: 0.1317

### per-class

- Away: 0.6750
- Bend: 1.0000
- Kneel: 0.0000
- Pick: 0.0000
- SStep: 0.0000
- Sit: 0.0000
- Towards: 0.0000

## val (10+24GHz source validation)

- acc: 0.9259
- macro-F1: 0.9258

## Frequency scan on 10GHz val

| f_virt (GHz) | acc | macro-F1 |
|---|---|---|
| 15.0 | 0.7436 | 0.7198 |
| 30.0 | 0.3462 | 0.2059 |
| 50.0 | 0.3077 | 0.1353 |
| 77.0 | 0.3077 | 0.1345 |
| 99.0 | 0.3077 | 0.1347 |
| 120.0 | 0.3077 | 0.1345 |
| 140.0 | 0.3077 | 0.1345 |

## Frequency scan on 24GHz val

| f_virt (GHz) | acc | macro-F1 |
|---|---|---|
| 15.0 | 0.6429 | 0.6125 |
| 30.0 | 0.9167 | 0.9149 |
| 50.0 | 0.5238 | 0.4654 |
| 77.0 | 0.3333 | 0.2069 |
| 99.0 | 0.2857 | 0.1581 |
| 120.0 | 0.2738 | 0.1226 |
| 140.0 | 0.2857 | 0.1452 |
