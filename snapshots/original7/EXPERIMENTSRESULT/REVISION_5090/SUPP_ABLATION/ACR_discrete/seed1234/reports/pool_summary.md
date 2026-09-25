# Pool selection summary

- threshold: source_val_acc_ema >= 0.90
- period: every 10 epochs
- pool size: 7
- selection metric: dev77_f1

## Members

| epoch | src_val_acc | src_val_F1 | dev77 acc | dev77 F1 | final77 acc | final77 F1 |
|---:|---:|---:|---:|---:|---:|---:|
|  40 | 0.9568 | 0.9567 | 0.5143 | 0.4619 | 0.5576 | 0.5074 |
|  50 | 0.9877 | 0.9878 | 0.7214 | 0.6993 | 0.7518 | 0.7228 |
|  60 | 0.9815 | 0.9818 | 0.7857 | 0.7874 | 0.8273 | 0.8174 |
|  70 | 0.9877 | 0.9878 | 0.8000 | 0.7985 | 0.8453 | 0.8405 |
|  80 | 0.9877 | 0.9878 | 0.7643 | 0.7599 | 0.8381 | 0.8344 |
|  90 | 0.9815 | 0.9816 | 0.7500 | 0.7446 | 0.8417 | 0.8388 |
| 100 | 0.9815 | 0.9816 | 0.7571 | 0.7513 | 0.8489 | 0.8458 |

## Selected (best in pool)

- epoch: **70**
- source val acc / F1 (EMA): 0.9877 / 0.9878
- 77GHz dev   acc / F1: 0.8000 / 0.7985
- 77GHz final acc / F1: 0.8453 / 0.8405
- checkpoint: `/scratch/st-zliu-1/heqingz/CrossCarrier_Fresh_20260922/CrossCarrier_Repro/EXPERIMENTSRESULT/REVISION_5090/SUPP_ABLATION/ACR_discrete/seed1234/checkpoints/best_pool_ema.pt`

### dev77 per-class F1
- Away: 0.9500
- Bend: 0.7000
- Kneel: 0.7500
- Pick: 0.8500
- SStep: 0.9500
- Sit: 0.6000
- Towards: 0.8000

### final77 per-class F1
- Away: 1.0000
- Bend: 0.7949
- Kneel: 0.8333
- Pick: 0.8750
- SStep: 1.0000
- Sit: 0.5676
- Towards: 0.8250
