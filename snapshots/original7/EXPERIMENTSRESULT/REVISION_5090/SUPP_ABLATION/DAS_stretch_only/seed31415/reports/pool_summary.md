# Pool selection summary

- threshold: source_val_acc_ema >= 0.90
- period: every 10 epochs
- pool size: 8
- selection metric: dev77_f1

## Members

| epoch | src_val_acc | src_val_F1 | dev77 acc | dev77 F1 | final77 acc | final77 F1 |
|---:|---:|---:|---:|---:|---:|---:|
|  30 | 0.9136 | 0.9138 | 0.3143 | 0.2371 | 0.3525 | 0.2719 |
|  40 | 0.9444 | 0.9453 | 0.4500 | 0.3940 | 0.4928 | 0.4175 |
|  50 | 0.9506 | 0.9518 | 0.6143 | 0.5750 | 0.6223 | 0.5789 |
|  60 | 0.9506 | 0.9518 | 0.6571 | 0.6223 | 0.6871 | 0.6468 |
|  70 | 0.9568 | 0.9576 | 0.6857 | 0.6480 | 0.7086 | 0.6713 |
|  80 | 0.9506 | 0.9513 | 0.6857 | 0.6429 | 0.7194 | 0.6834 |
|  90 | 0.9506 | 0.9513 | 0.6857 | 0.6326 | 0.7194 | 0.6871 |
| 100 | 0.9444 | 0.9450 | 0.6929 | 0.6374 | 0.7302 | 0.6971 |

## Selected (best in pool)

- epoch: **70**
- source val acc / F1 (EMA): 0.9568 / 0.9576
- 77GHz dev   acc / F1: 0.6857 / 0.6480
- 77GHz final acc / F1: 0.7086 / 0.6713
- checkpoint: `/scratch/st-zliu-1/heqingz/CrossCarrier_Fresh_20260922/CrossCarrier_Repro/EXPERIMENTSRESULT/REVISION_5090/SUPP_ABLATION/DAS_stretch_only/seed31415/checkpoints/best_pool_ema.pt`

### dev77 per-class F1
- Away: 0.8500
- Bend: 0.7000
- Kneel: 0.8500
- Pick: 0.7000
- SStep: 0.9500
- Sit: 0.0500
- Towards: 0.7000

### final77 per-class F1
- Away: 0.8500
- Bend: 0.7436
- Kneel: 0.8810
- Pick: 0.7250
- SStep: 0.8750
- Sit: 0.0811
- Towards: 0.7500
