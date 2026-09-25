# Pool selection summary

- threshold: source_val_acc_ema >= 0.90
- period: every 10 epochs
- pool size: 7
- selection metric: dev77_f1

## Members

| epoch | src_val_acc | src_val_F1 | dev77 acc | dev77 F1 | final77 acc | final77 F1 |
|---:|---:|---:|---:|---:|---:|---:|
|  40 | 0.9259 | 0.9278 | 0.5714 | 0.5102 | 0.6187 | 0.5636 |
|  50 | 0.9321 | 0.9332 | 0.7000 | 0.6964 | 0.7806 | 0.7611 |
|  60 | 0.9444 | 0.9458 | 0.7643 | 0.7586 | 0.8237 | 0.8183 |
|  70 | 0.9630 | 0.9634 | 0.8071 | 0.8046 | 0.8561 | 0.8545 |
|  80 | 0.9630 | 0.9634 | 0.8071 | 0.8046 | 0.8741 | 0.8717 |
|  90 | 0.9691 | 0.9694 | 0.7929 | 0.7884 | 0.8633 | 0.8585 |
| 100 | 0.9753 | 0.9755 | 0.8000 | 0.7952 | 0.8597 | 0.8542 |

## Selected (best in pool)

- epoch: **70**
- source val acc / F1 (EMA): 0.9630 / 0.9634
- 77GHz dev   acc / F1: 0.8071 / 0.8046
- 77GHz final acc / F1: 0.8561 / 0.8545
- checkpoint: `/scratch/st-zliu-1/heqingz/CrossCarrier_Fresh_20260922/CrossCarrier_Repro/EXPERIMENTSRESULT/REVISION_5090/SUPP_ABLATION/ACR_Lfreq_only/seed31415/checkpoints/best_pool_ema.pt`

### dev77 per-class F1
- Away: 0.9000
- Bend: 0.6500
- Kneel: 0.7500
- Pick: 0.9000
- SStep: 1.0000
- Sit: 0.6500
- Towards: 0.8000

### final77 per-class F1
- Away: 0.9250
- Bend: 0.7436
- Kneel: 0.7857
- Pick: 0.9000
- SStep: 1.0000
- Sit: 0.7297
- Towards: 0.9000
