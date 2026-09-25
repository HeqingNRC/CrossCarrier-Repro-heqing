# Pool selection summary

- threshold: source_val_acc_ema >= 0.90
- period: every 10 epochs
- pool size: 7
- selection metric: dev77_f1

## Members

| epoch | src_val_acc | src_val_F1 | dev77 acc | dev77 F1 | final77 acc | final77 F1 |
|---:|---:|---:|---:|---:|---:|---:|
|  40 | 0.9321 | 0.9344 | 0.5929 | 0.5475 | 0.6259 | 0.5756 |
|  50 | 0.9506 | 0.9502 | 0.6714 | 0.6683 | 0.7518 | 0.7391 |
|  60 | 0.9630 | 0.9627 | 0.7357 | 0.7351 | 0.8022 | 0.7990 |
|  70 | 0.9630 | 0.9621 | 0.7429 | 0.7408 | 0.8022 | 0.8005 |
|  80 | 0.9630 | 0.9621 | 0.7357 | 0.7329 | 0.7950 | 0.7912 |
|  90 | 0.9691 | 0.9685 | 0.7286 | 0.7253 | 0.7950 | 0.7919 |
| 100 | 0.9691 | 0.9685 | 0.7214 | 0.7124 | 0.8022 | 0.7986 |

## Selected (best in pool)

- epoch: **70**
- source val acc / F1 (EMA): 0.9630 / 0.9621
- 77GHz dev   acc / F1: 0.7429 / 0.7408
- 77GHz final acc / F1: 0.8022 / 0.8005
- checkpoint: `/scratch/st-zliu-1/heqingz/CrossCarrier_Fresh_20260922/CrossCarrier_Repro/EXPERIMENTSRESULT/REVISION_5090/SUPP_ABLATION/ACR_Lfreq_only/seed42/checkpoints/best_pool_ema.pt`

### dev77 per-class F1
- Away: 0.9500
- Bend: 0.5500
- Kneel: 0.7500
- Pick: 0.7500
- SStep: 1.0000
- Sit: 0.6000
- Towards: 0.6000

### final77 per-class F1
- Away: 1.0000
- Bend: 0.6667
- Kneel: 0.7857
- Pick: 0.8500
- SStep: 1.0000
- Sit: 0.6216
- Towards: 0.6750
