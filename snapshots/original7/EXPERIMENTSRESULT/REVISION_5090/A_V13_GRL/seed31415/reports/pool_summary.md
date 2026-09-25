# Pool selection summary

- threshold: source_val_acc_ema >= 0.90
- period: every 10 epochs
- pool size: 7
- selection metric: dev77_f1

## Members

| epoch | src_val_acc | src_val_F1 | dev77 acc | dev77 F1 | final77 acc | final77 F1 |
|---:|---:|---:|---:|---:|---:|---:|
|  40 | 0.9444 | 0.9455 | 0.5357 | 0.4535 | 0.5863 | 0.5099 |
|  50 | 0.9568 | 0.9574 | 0.6786 | 0.6353 | 0.7050 | 0.6501 |
|  60 | 0.9691 | 0.9697 | 0.7571 | 0.7401 | 0.8094 | 0.7969 |
|  70 | 0.9630 | 0.9635 | 0.8286 | 0.8234 | 0.8561 | 0.8522 |
|  80 | 0.9630 | 0.9635 | 0.8571 | 0.8539 | 0.8669 | 0.8637 |
|  90 | 0.9630 | 0.9635 | 0.8429 | 0.8403 | 0.8813 | 0.8785 |
| 100 | 0.9630 | 0.9635 | 0.8500 | 0.8470 | 0.8921 | 0.8902 |

## Selected (best in pool)

- epoch: **80**
- source val acc / F1 (EMA): 0.9630 / 0.9635
- 77GHz dev   acc / F1: 0.8571 / 0.8539
- 77GHz final acc / F1: 0.8669 / 0.8637
- checkpoint: `EXPERIMENTSRESULT\REVISION_5090\A_V13_GRL\seed31415\checkpoints\best_pool_ema.pt`

### dev77 per-class F1
- Away: 1.0000
- Bend: 0.8500
- Kneel: 0.9000
- Pick: 0.8000
- SStep: 1.0000
- Sit: 0.6000
- Towards: 0.8500

### final77 per-class F1
- Away: 1.0000
- Bend: 0.8462
- Kneel: 0.8810
- Pick: 0.8500
- SStep: 1.0000
- Sit: 0.6486
- Towards: 0.8250
