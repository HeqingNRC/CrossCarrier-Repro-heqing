# Pool selection summary

- threshold: source_val_acc_ema >= 0.90
- period: every 10 epochs
- pool size: 7
- selection metric: dev77_f1

## Members

| epoch | src_val_acc | src_val_F1 | dev77 acc | dev77 F1 | final77 acc | final77 F1 |
|---:|---:|---:|---:|---:|---:|---:|
|  40 | 0.9321 | 0.9334 | 0.5000 | 0.4244 | 0.5108 | 0.4454 |
|  50 | 0.9568 | 0.9574 | 0.6286 | 0.5865 | 0.6547 | 0.6107 |
|  60 | 0.9568 | 0.9574 | 0.7357 | 0.7260 | 0.7770 | 0.7685 |
|  70 | 0.9568 | 0.9574 | 0.7786 | 0.7738 | 0.8201 | 0.8182 |
|  80 | 0.9630 | 0.9634 | 0.8071 | 0.8039 | 0.8381 | 0.8353 |
|  90 | 0.9691 | 0.9694 | 0.8071 | 0.8020 | 0.8417 | 0.8389 |
| 100 | 0.9691 | 0.9694 | 0.8000 | 0.7934 | 0.8453 | 0.8423 |

## Selected (best in pool)

- epoch: **80**
- source val acc / F1 (EMA): 0.9630 / 0.9634
- 77GHz dev   acc / F1: 0.8071 / 0.8039
- 77GHz final acc / F1: 0.8381 / 0.8353
- checkpoint: `/scratch/st-zliu-1/heqingz/CrossCarrier_Fresh_20260922/CrossCarrier_Repro/EXPERIMENTSRESULT/REVISION_5090/SUPP_ABLATION/ACR_discrete/seed31415/checkpoints/best_pool_ema.pt`

### dev77 per-class F1
- Away: 0.9500
- Bend: 0.6500
- Kneel: 0.8000
- Pick: 0.8500
- SStep: 1.0000
- Sit: 0.6000
- Towards: 0.8000

### final77 per-class F1
- Away: 1.0000
- Bend: 0.7949
- Kneel: 0.8095
- Pick: 0.7750
- SStep: 1.0000
- Sit: 0.6216
- Towards: 0.8500
