# V20-V15V13-KnownPeople-UnknownFreq

- Generated: 2026-09-24 00:50:19
- Backbone: vit_large_patch16_dinov3.lvd1689m
- Adapter mode: lora
- Checkpoint: vit_large_patch16_dinov3_lvd1689m_best_sourcequalified77_ema.pt (sourcequalified)
- Selection: source-qualified diagnostic selected (source-val ema acc >= 0.90, then best 77GHz ema macro-F1)

## 77GHz final test

- acc: 0.8993
- macro-F1: 0.8998

### per-class

- Away: 1.0000
- Bend: 0.8205
- Kneel: 0.8810
- Pick: 0.8750
- SStep: 1.0000
- Sit: 0.7838
- Towards: 0.9250

## val (10+24GHz source validation)

- acc: 0.9691
- macro-F1: 0.9687

## Frequency scan on 10GHz val

| f_virt (GHz) | acc | macro-F1 |
|---|---|---|
| 15.0 | 0.9744 | 0.9756 |
| 30.0 | 0.9744 | 0.9760 |
| 50.0 | 0.9487 | 0.9517 |
| 77.0 | 0.8846 | 0.8851 |
| 99.0 | 0.8205 | 0.8178 |
| 120.0 | 0.7564 | 0.7568 |
| 140.0 | 0.6795 | 0.6741 |

## Frequency scan on 24GHz val

| f_virt (GHz) | acc | macro-F1 |
|---|---|---|
| 15.0 | 0.9881 | 0.9875 |
| 30.0 | 0.9762 | 0.9751 |
| 50.0 | 0.9643 | 0.9646 |
| 77.0 | 0.9643 | 0.9646 |
| 99.0 | 0.8810 | 0.8802 |
| 120.0 | 0.8095 | 0.8005 |
| 140.0 | 0.7738 | 0.7583 |
