# Pool selection summary

- threshold: source_val_acc_ema >= 0.90
- period: every 10 epochs
- pool size: 7
- selection metric: dev77_f1

## Members

| epoch | src_val_acc | src_val_F1 | dev77 acc | dev77 F1 | final77 acc | final77 F1 |
|---:|---:|---:|---:|---:|---:|---:|
|  40 | 0.9444 | 0.9455 | 0.6000 | 0.5530 | 0.6295 | 0.5705 |
|  50 | 0.9753 | 0.9758 | 0.6929 | 0.6605 | 0.7050 | 0.6765 |
|  60 | 0.9815 | 0.9819 | 0.7214 | 0.7048 | 0.7806 | 0.7707 |
|  70 | 0.9815 | 0.9811 | 0.7357 | 0.7283 | 0.7950 | 0.7845 |
|  80 | 0.9753 | 0.9749 | 0.7286 | 0.7188 | 0.8129 | 0.8062 |
|  90 | 0.9753 | 0.9749 | 0.7357 | 0.7264 | 0.8165 | 0.8089 |
| 100 | 0.9753 | 0.9749 | 0.7429 | 0.7351 | 0.8237 | 0.8168 |

## Selected (best in pool)

- epoch: **100**
- source val acc / F1 (EMA): 0.9753 / 0.9749
- 77GHz dev   acc / F1: 0.7429 / 0.7351
- 77GHz final acc / F1: 0.8237 / 0.8168
- checkpoint: `EXPERIMENTSRESULT\REVISION_5090\GRL_TUNE\w0.3_seed42\checkpoints\best_pool_ema.pt`

### dev77 per-class F1
- Away: 0.9500
- Bend: 0.5500
- Kneel: 0.9000
- Pick: 0.7000
- SStep: 1.0000
- Sit: 0.5500
- Towards: 0.5500

### final77 per-class F1
- Away: 0.9750
- Bend: 0.7949
- Kneel: 0.8571
- Pick: 0.8750
- SStep: 1.0000
- Sit: 0.4595
- Towards: 0.7750
