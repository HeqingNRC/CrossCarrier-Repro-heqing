# Pool selection summary

- threshold: source_val_acc_ema >= 0.90
- period: every 10 epochs
- pool size: 7
- selection metric: dev77_f1

## Members

| epoch | src_val_acc | src_val_F1 | dev77 acc | dev77 F1 | final77 acc | final77 F1 |
|---:|---:|---:|---:|---:|---:|---:|
|  40 | 0.9630 | 0.9635 | 0.5357 | 0.4766 | 0.5863 | 0.5176 |
|  50 | 0.9568 | 0.9574 | 0.7000 | 0.6542 | 0.7230 | 0.6742 |
|  60 | 0.9568 | 0.9571 | 0.7357 | 0.7050 | 0.7734 | 0.7431 |
|  70 | 0.9630 | 0.9631 | 0.7071 | 0.6733 | 0.7878 | 0.7607 |
|  80 | 0.9630 | 0.9631 | 0.6929 | 0.6595 | 0.7770 | 0.7519 |
|  90 | 0.9630 | 0.9631 | 0.6643 | 0.6283 | 0.7554 | 0.7284 |
| 100 | 0.9630 | 0.9631 | 0.6500 | 0.6107 | 0.7374 | 0.7086 |

## Selected (best in pool)

- epoch: **60**
- source val acc / F1 (EMA): 0.9568 / 0.9571
- 77GHz dev   acc / F1: 0.7357 / 0.7050
- 77GHz final acc / F1: 0.7734 / 0.7431
- checkpoint: `/scratch/st-zliu-1/heqingz/CrossCarrier_Fresh_20260922/CrossCarrier_Repro/EXPERIMENTSRESULT/REVISION_5090/SUPP_ABLATION/ACR_GRL_only/seed42/checkpoints/best_pool_ema.pt`

### dev77 per-class F1
- Away: 0.9500
- Bend: 0.6000
- Kneel: 0.9500
- Pick: 0.6500
- SStep: 0.9000
- Sit: 0.1000
- Towards: 1.0000

### final77 per-class F1
- Away: 0.9750
- Bend: 0.7949
- Kneel: 0.8810
- Pick: 0.7000
- SStep: 0.9750
- Sit: 0.0811
- Towards: 0.9500
