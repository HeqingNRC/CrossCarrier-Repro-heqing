# Pool selection summary

- threshold: source_val_acc_ema >= 0.90
- period: every 10 epochs
- pool size: 7
- selection metric: dev77_f1

## Members

| epoch | src_val_acc | src_val_F1 | dev77 acc | dev77 F1 | final77 acc | final77 F1 |
|---:|---:|---:|---:|---:|---:|---:|
|  40 | 0.9630 | 0.9642 | 0.5500 | 0.4827 | 0.5935 | 0.5401 |
|  50 | 0.9630 | 0.9636 | 0.6929 | 0.6558 | 0.7662 | 0.7191 |
|  60 | 0.9630 | 0.9627 | 0.7429 | 0.7303 | 0.8129 | 0.7937 |
|  70 | 0.9630 | 0.9627 | 0.8071 | 0.8037 | 0.8381 | 0.8251 |
|  80 | 0.9630 | 0.9627 | 0.8000 | 0.7961 | 0.8561 | 0.8502 |
|  90 | 0.9630 | 0.9628 | 0.7857 | 0.7811 | 0.8633 | 0.8606 |
| 100 | 0.9630 | 0.9628 | 0.7857 | 0.7811 | 0.8597 | 0.8575 |

## Selected (best in pool)

- epoch: **70**
- source val acc / F1 (EMA): 0.9630 / 0.9627
- 77GHz dev   acc / F1: 0.8071 / 0.8037
- 77GHz final acc / F1: 0.8381 / 0.8251
- checkpoint: `EXPERIMENTSRESULT\REVISION_5090\A_V13_GRL\seed1234\checkpoints\best_pool_ema.pt`

### dev77 per-class F1
- Away: 0.9500
- Bend: 0.7000
- Kneel: 0.8500
- Pick: 0.7500
- SStep: 1.0000
- Sit: 0.6000
- Towards: 0.8000

### final77 per-class F1
- Away: 1.0000
- Bend: 0.8718
- Kneel: 0.8571
- Pick: 0.8250
- SStep: 1.0000
- Sit: 0.3784
- Towards: 0.9000
