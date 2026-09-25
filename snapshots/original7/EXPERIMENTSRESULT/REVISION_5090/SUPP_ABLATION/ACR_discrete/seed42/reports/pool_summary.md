# Pool selection summary

- threshold: source_val_acc_ema >= 0.90
- period: every 10 epochs
- pool size: 7
- selection metric: dev77_f1

## Members

| epoch | src_val_acc | src_val_F1 | dev77 acc | dev77 F1 | final77 acc | final77 F1 |
|---:|---:|---:|---:|---:|---:|---:|
|  40 | 0.9630 | 0.9635 | 0.6143 | 0.5433 | 0.6475 | 0.5644 |
|  50 | 0.9691 | 0.9695 | 0.7000 | 0.6547 | 0.7230 | 0.6676 |
|  60 | 0.9691 | 0.9694 | 0.7000 | 0.6648 | 0.7806 | 0.7529 |
|  70 | 0.9691 | 0.9694 | 0.7357 | 0.7097 | 0.7770 | 0.7513 |
|  80 | 0.9691 | 0.9694 | 0.7357 | 0.7138 | 0.7698 | 0.7483 |
|  90 | 0.9691 | 0.9694 | 0.7357 | 0.7184 | 0.7734 | 0.7516 |
| 100 | 0.9691 | 0.9694 | 0.7286 | 0.7098 | 0.7770 | 0.7573 |

## Selected (best in pool)

- epoch: **90**
- source val acc / F1 (EMA): 0.9691 / 0.9694
- 77GHz dev   acc / F1: 0.7357 / 0.7184
- 77GHz final acc / F1: 0.7734 / 0.7516
- checkpoint: `/scratch/st-zliu-1/heqingz/CrossCarrier_Fresh_20260922/CrossCarrier_Repro/EXPERIMENTSRESULT/REVISION_5090/SUPP_ABLATION/ACR_discrete/seed42/checkpoints/best_pool_ema.pt`

### dev77 per-class F1
- Away: 0.9000
- Bend: 0.5500
- Kneel: 0.9000
- Pick: 0.7500
- SStep: 1.0000
- Sit: 0.2000
- Towards: 0.8500

### final77 per-class F1
- Away: 0.9750
- Bend: 0.6667
- Kneel: 0.8095
- Pick: 0.8750
- SStep: 1.0000
- Sit: 0.1892
- Towards: 0.8500
