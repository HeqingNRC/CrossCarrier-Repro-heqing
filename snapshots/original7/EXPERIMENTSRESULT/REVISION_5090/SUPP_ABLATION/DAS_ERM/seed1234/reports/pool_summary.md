# Pool selection summary

- threshold: source_val_acc_ema >= 0.90
- period: every 10 epochs
- pool size: 6
- selection metric: dev77_f1

## Members

| epoch | src_val_acc | src_val_F1 | dev77 acc | dev77 F1 | final77 acc | final77 F1 |
|---:|---:|---:|---:|---:|---:|---:|
|  50 | 0.9321 | 0.9350 | 0.1429 | 0.0357 | 0.1403 | 0.0352 |
|  60 | 0.9383 | 0.9404 | 0.1429 | 0.0357 | 0.1403 | 0.0352 |
|  70 | 0.9444 | 0.9463 | 0.1571 | 0.0621 | 0.1439 | 0.0422 |
|  80 | 0.9506 | 0.9521 | 0.1643 | 0.0737 | 0.1475 | 0.0490 |
|  90 | 0.9568 | 0.9572 | 0.1786 | 0.0992 | 0.1511 | 0.0559 |
| 100 | 0.9506 | 0.9509 | 0.1857 | 0.1103 | 0.1619 | 0.0740 |

## Selected (best in pool)

- epoch: **100**
- source val acc / F1 (EMA): 0.9506 / 0.9509
- 77GHz dev   acc / F1: 0.1857 / 0.1103
- 77GHz final acc / F1: 0.1619 / 0.0740
- checkpoint: `/scratch/st-zliu-1/heqingz/CrossCarrier_Fresh_20260922/CrossCarrier_Repro/EXPERIMENTSRESULT/REVISION_5090/SUPP_ABLATION/DAS_ERM/seed1234/checkpoints/best_pool_ema.pt`

### dev77 per-class F1
- Away: 0.1500
- Bend: 1.0000
- Kneel: 0.0000
- Pick: 0.0000
- SStep: 0.0000
- Sit: 0.0000
- Towards: 0.1500

### final77 per-class F1
- Away: 0.0250
- Bend: 1.0000
- Kneel: 0.0000
- Pick: 0.0000
- SStep: 0.0000
- Sit: 0.0000
- Towards: 0.1250
