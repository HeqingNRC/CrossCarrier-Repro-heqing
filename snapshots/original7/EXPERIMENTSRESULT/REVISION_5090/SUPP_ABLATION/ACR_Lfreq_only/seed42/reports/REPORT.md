# V20-V15V13-KnownPeople-UnknownFreq

- Generated: 2026-09-23 18:01:30
- Backbone: vit_large_patch16_dinov3.lvd1689m
- Adapter mode: lora
- Checkpoint: vit_large_patch16_dinov3_lvd1689m_best_sourcequalified77_ema.pt (sourcequalified)
- Selection: source-qualified diagnostic selected (source-val ema acc >= 0.90, then best 77GHz ema macro-F1)

## 77GHz final test

- acc: 0.8094
- macro-F1: 0.8097

### per-class

- Away: 0.9750
- Bend: 0.7436
- Kneel: 0.7857
- Pick: 0.8500
- SStep: 1.0000
- Sit: 0.5946
- Towards: 0.7000

## val (10+24GHz source validation)

- acc: 0.9568
- macro-F1: 0.9561

## Frequency scan on 10GHz val

| f_virt (GHz) | acc | macro-F1 |
|---|---|---|
| 15.0 | 0.9615 | 0.9593 |
| 30.0 | 0.9872 | 0.9881 |
| 50.0 | 0.9872 | 0.9881 |
| 77.0 | 0.9615 | 0.9642 |
| 99.0 | 0.9615 | 0.9642 |
| 120.0 | 0.9231 | 0.9221 |
| 140.0 | 0.9231 | 0.9221 |

## Frequency scan on 24GHz val

| f_virt (GHz) | acc | macro-F1 |
|---|---|---|
| 15.0 | 0.9881 | 0.9876 |
| 30.0 | 0.9881 | 0.9875 |
| 50.0 | 0.9881 | 0.9875 |
| 77.0 | 0.9881 | 0.9875 |
| 99.0 | 0.9881 | 0.9875 |
| 120.0 | 0.9881 | 0.9875 |
| 140.0 | 0.9524 | 0.9534 |
