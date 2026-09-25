# Pool selection summary

- threshold: source_val_acc_ema >= 0.90
- period: every 10 epochs
- pool size: 7
- selection metric: dev77_f1

## Members

| epoch | src_val_acc | src_val_F1 | dev77 acc | dev77 F1 | final77 acc | final77 F1 |
|---:|---:|---:|---:|---:|---:|---:|
|  40 | 0.9074 | 0.9072 | 0.4000 | 0.3335 | 0.4460 | 0.3641 |
|  50 | 0.9568 | 0.9563 | 0.4929 | 0.4231 | 0.5863 | 0.5095 |
|  60 | 0.9630 | 0.9626 | 0.5571 | 0.4847 | 0.6079 | 0.5349 |
|  70 | 0.9630 | 0.9624 | 0.5643 | 0.5012 | 0.6223 | 0.5539 |
|  80 | 0.9691 | 0.9692 | 0.5643 | 0.5045 | 0.6475 | 0.5892 |
|  90 | 0.9630 | 0.9634 | 0.5714 | 0.5171 | 0.6475 | 0.5853 |
| 100 | 0.9691 | 0.9697 | 0.5857 | 0.5432 | 0.6295 | 0.5741 |

## Selected (best in pool)

- epoch: **100**
- source val acc / F1 (EMA): 0.9691 / 0.9697
- 77GHz dev   acc / F1: 0.5857 / 0.5432
- 77GHz final acc / F1: 0.6295 / 0.5741
- checkpoint: `/scratch/st-zliu-1/heqingz/CrossCarrier_Fresh_20260922/CrossCarrier_Repro/EXPERIMENTSRESULT/REVISION_5090/SUPP_ABLATION/SCHED_fixedfull_ACR/seed31415/checkpoints/best_pool_ema.pt`

### dev77 per-class F1
- Away: 0.0500
- Bend: 0.5500
- Kneel: 0.7500
- Pick: 0.8000
- SStep: 1.0000
- Sit: 0.2000
- Towards: 0.7500

### final77 per-class F1
- Away: 0.0000
- Bend: 0.6923
- Kneel: 0.8571
- Pick: 0.7500
- SStep: 0.9500
- Sit: 0.2432
- Towards: 0.8750
