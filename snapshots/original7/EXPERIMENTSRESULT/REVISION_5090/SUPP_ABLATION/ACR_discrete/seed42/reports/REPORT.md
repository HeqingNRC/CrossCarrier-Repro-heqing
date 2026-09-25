# V20-V15V13-KnownPeople-UnknownFreq

- Generated: 2026-09-23 20:32:19
- Backbone: vit_large_patch16_dinov3.lvd1689m
- Adapter mode: lora
- Checkpoint: vit_large_patch16_dinov3_lvd1689m_best_sourcequalified77_ema.pt (sourcequalified)
- Selection: source-qualified diagnostic selected (source-val ema acc >= 0.90, then best 77GHz ema macro-F1)

## 77GHz final test

- acc: 0.7842
- macro-F1: 0.7593

### per-class

- Away: 0.9750
- Bend: 0.7692
- Kneel: 0.8333
- Pick: 0.8250
- SStep: 1.0000
- Sit: 0.1351
- Towards: 0.9000

## val (10+24GHz source validation)

- acc: 0.9691
- macro-F1: 0.9694

## Frequency scan on 10GHz val

| f_virt (GHz) | acc | macro-F1 |
|---|---|---|
| 15.0 | 0.9872 | 0.9881 |
| 30.0 | 0.9872 | 0.9881 |
| 50.0 | 0.9872 | 0.9881 |
| 77.0 | 0.9744 | 0.9760 |
| 99.0 | 0.9487 | 0.9523 |
| 120.0 | 0.9231 | 0.9270 |
| 140.0 | 0.8974 | 0.8999 |

## Frequency scan on 24GHz val

| f_virt (GHz) | acc | macro-F1 |
|---|---|---|
| 15.0 | 0.9881 | 0.9876 |
| 30.0 | 0.9643 | 0.9627 |
| 50.0 | 0.9881 | 0.9875 |
| 77.0 | 0.9762 | 0.9751 |
| 99.0 | 0.9762 | 0.9756 |
| 120.0 | 0.9643 | 0.9646 |
| 140.0 | 0.9167 | 0.9173 |
