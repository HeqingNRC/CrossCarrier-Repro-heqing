# Pool selection summary

- threshold: source_val_acc_ema >= 0.90
- period: every 10 epochs
- pool size: 7
- selection metric: dev77_f1

## Members

| epoch | src_val_acc | src_val_F1 | dev77 acc | dev77 F1 | final77 acc | final77 F1 |
|---:|---:|---:|---:|---:|---:|---:|
|  40 | 0.9012 | 0.8972 | 0.3000 | 0.2110 | 0.3058 | 0.2223 |
|  50 | 0.9383 | 0.9395 | 0.4571 | 0.3994 | 0.4928 | 0.4193 |
|  60 | 0.9568 | 0.9579 | 0.5857 | 0.5558 | 0.6007 | 0.5642 |
|  70 | 0.9630 | 0.9640 | 0.6357 | 0.6127 | 0.6403 | 0.6115 |
|  80 | 0.9630 | 0.9640 | 0.6286 | 0.6100 | 0.6583 | 0.6345 |
|  90 | 0.9630 | 0.9640 | 0.6500 | 0.6319 | 0.6763 | 0.6559 |
| 100 | 0.9630 | 0.9640 | 0.6571 | 0.6379 | 0.6835 | 0.6626 |

## Selected (best in pool)

- epoch: **100**
- source val acc / F1 (EMA): 0.9630 / 0.9640
- 77GHz dev   acc / F1: 0.6571 / 0.6379
- 77GHz final acc / F1: 0.6835 / 0.6626
- checkpoint: `/scratch/st-zliu-1/heqingz/CrossCarrier_Fresh_20260922/CrossCarrier_Repro/EXPERIMENTSRESULT/REVISION_5090/SUPP_ABLATION/DAS_stretch_only/seed42/checkpoints/best_pool_ema.pt`

### dev77 per-class F1
- Away: 0.9500
- Bend: 0.5500
- Kneel: 0.8000
- Pick: 0.7000
- SStep: 0.9000
- Sit: 0.1500
- Towards: 0.5500

### final77 per-class F1
- Away: 1.0000
- Bend: 0.6667
- Kneel: 0.7857
- Pick: 0.7250
- SStep: 0.7250
- Sit: 0.1081
- Towards: 0.7250
