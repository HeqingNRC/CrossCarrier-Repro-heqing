# V20-V15V13-KnownPeople-UnknownFreq

- Generated: 2026-09-23 19:19:44
- Backbone: vit_large_patch16_dinov3.lvd1689m
- Adapter mode: lora
- Checkpoint: vit_large_patch16_dinov3_lvd1689m_best_sourcequalified77_ema.pt (sourcequalified)
- Selection: source-qualified diagnostic selected (source-val ema acc >= 0.90, then best 77GHz ema macro-F1)

## 77GHz final test

- acc: 0.7914
- macro-F1: 0.7644

### per-class

- Away: 0.9750
- Bend: 0.7949
- Kneel: 0.8810
- Pick: 0.8000
- SStep: 1.0000
- Sit: 0.1622
- Towards: 0.8750

## val (10+24GHz source validation)

- acc: 0.9630
- macro-F1: 0.9631

## Frequency scan on 10GHz val

| f_virt (GHz) | acc | macro-F1 |
|---|---|---|
| 15.0 | 0.9872 | 0.9881 |
| 30.0 | 0.9744 | 0.9760 |
| 50.0 | 0.9744 | 0.9760 |
| 77.0 | 0.9487 | 0.9474 |
| 99.0 | 0.9231 | 0.9229 |
| 120.0 | 0.9231 | 0.9221 |
| 140.0 | 0.9103 | 0.9107 |

## Frequency scan on 24GHz val

| f_virt (GHz) | acc | macro-F1 |
|---|---|---|
| 15.0 | 0.9762 | 0.9756 |
| 30.0 | 0.9762 | 0.9751 |
| 50.0 | 0.9762 | 0.9760 |
| 77.0 | 0.9762 | 0.9751 |
| 99.0 | 0.9762 | 0.9751 |
| 120.0 | 0.9643 | 0.9651 |
| 140.0 | 0.9524 | 0.9531 |
