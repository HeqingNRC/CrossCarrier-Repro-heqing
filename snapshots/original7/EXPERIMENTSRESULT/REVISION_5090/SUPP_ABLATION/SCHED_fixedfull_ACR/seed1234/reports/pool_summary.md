# Pool selection summary

- threshold: source_val_acc_ema >= 0.90
- period: every 10 epochs
- pool size: 6
- selection metric: dev77_f1

## Members

| epoch | src_val_acc | src_val_F1 | dev77 acc | dev77 F1 | final77 acc | final77 F1 |
|---:|---:|---:|---:|---:|---:|---:|
|  50 | 0.9383 | 0.9373 | 0.6000 | 0.5446 | 0.6439 | 0.5708 |
|  60 | 0.9383 | 0.9369 | 0.6286 | 0.5816 | 0.6906 | 0.6362 |
|  70 | 0.9444 | 0.9444 | 0.6500 | 0.6069 | 0.7014 | 0.6493 |
|  80 | 0.9506 | 0.9502 | 0.6571 | 0.6163 | 0.7014 | 0.6573 |
|  90 | 0.9568 | 0.9564 | 0.6214 | 0.5782 | 0.6942 | 0.6502 |
| 100 | 0.9506 | 0.9500 | 0.5857 | 0.5480 | 0.6799 | 0.6394 |

## Selected (best in pool)

- epoch: **80**
- source val acc / F1 (EMA): 0.9506 / 0.9502
- 77GHz dev   acc / F1: 0.6571 / 0.6163
- 77GHz final acc / F1: 0.7014 / 0.6573
- checkpoint: `/scratch/st-zliu-1/heqingz/CrossCarrier_Fresh_20260922/CrossCarrier_Repro/EXPERIMENTSRESULT/REVISION_5090/SUPP_ABLATION/SCHED_fixedfull_ACR/seed1234/checkpoints/best_pool_ema.pt`

### dev77 per-class F1
- Away: 0.0000
- Bend: 0.6500
- Kneel: 0.8500
- Pick: 0.9000
- SStep: 0.9500
- Sit: 0.4500
- Towards: 0.8000

### final77 per-class F1
- Away: 0.0250
- Bend: 0.7692
- Kneel: 0.9048
- Pick: 0.9000
- SStep: 1.0000
- Sit: 0.4324
- Towards: 0.8500
