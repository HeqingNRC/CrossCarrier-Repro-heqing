# Pool selection summary

- threshold: source_val_acc_ema >= 0.90
- period: every 10 epochs
- pool size: 7
- selection metric: dev77_f1

## Members

| epoch | src_val_acc | src_val_F1 | dev77 acc | dev77 F1 | final77 acc | final77 F1 |
|---:|---:|---:|---:|---:|---:|---:|
|  40 | 0.9321 | 0.9346 | 0.2786 | 0.2106 | 0.2842 | 0.2112 |
|  50 | 0.9691 | 0.9701 | 0.4214 | 0.3901 | 0.4101 | 0.3771 |
|  60 | 0.9753 | 0.9761 | 0.5714 | 0.5587 | 0.5935 | 0.6077 |
|  70 | 0.9753 | 0.9749 | 0.6429 | 0.6380 | 0.6763 | 0.6819 |
|  80 | 0.9753 | 0.9749 | 0.6500 | 0.6381 | 0.6942 | 0.6939 |
|  90 | 0.9753 | 0.9749 | 0.6500 | 0.6346 | 0.7302 | 0.7318 |
| 100 | 0.9753 | 0.9749 | 0.6429 | 0.6276 | 0.7302 | 0.7310 |

## Selected (best in pool)

- epoch: **80**
- source val acc / F1 (EMA): 0.9753 / 0.9749
- 77GHz dev   acc / F1: 0.6500 / 0.6381
- 77GHz final acc / F1: 0.6942 / 0.6939
- checkpoint: `/scratch/st-zliu-1/heqingz/CrossCarrier_Fresh_20260922/CrossCarrier_Repro/EXPERIMENTSRESULT/REVISION_5090/SUPP_ABLATION/DAS_stretch_only/seed1234/checkpoints/best_pool_ema.pt`

### dev77 per-class F1
- Away: 0.9500
- Bend: 1.0000
- Kneel: 0.4500
- Pick: 0.2500
- SStep: 0.9500
- Sit: 0.3500
- Towards: 0.6000

### final77 per-class F1
- Away: 0.9750
- Bend: 0.9487
- Kneel: 0.5952
- Pick: 0.2500
- SStep: 0.9750
- Sit: 0.4865
- Towards: 0.6250
