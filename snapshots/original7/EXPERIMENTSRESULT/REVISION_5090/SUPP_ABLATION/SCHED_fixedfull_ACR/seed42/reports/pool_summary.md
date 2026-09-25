# Pool selection summary

- threshold: source_val_acc_ema >= 0.90
- period: every 10 epochs
- pool size: 6
- selection metric: dev77_f1

## Members

| epoch | src_val_acc | src_val_F1 | dev77 acc | dev77 F1 | final77 acc | final77 F1 |
|---:|---:|---:|---:|---:|---:|---:|
|  50 | 0.9691 | 0.9694 | 0.5071 | 0.4624 | 0.5540 | 0.5148 |
|  60 | 0.9630 | 0.9628 | 0.5500 | 0.5269 | 0.5827 | 0.5520 |
|  70 | 0.9630 | 0.9628 | 0.5500 | 0.5294 | 0.6043 | 0.5873 |
|  80 | 0.9630 | 0.9628 | 0.5571 | 0.5390 | 0.6187 | 0.6026 |
|  90 | 0.9630 | 0.9628 | 0.5571 | 0.5356 | 0.6151 | 0.6003 |
| 100 | 0.9630 | 0.9628 | 0.5643 | 0.5410 | 0.6115 | 0.5965 |

## Selected (best in pool)

- epoch: **100**
- source val acc / F1 (EMA): 0.9630 / 0.9628
- 77GHz dev   acc / F1: 0.5643 / 0.5410
- 77GHz final acc / F1: 0.6115 / 0.5965
- checkpoint: `/scratch/st-zliu-1/heqingz/CrossCarrier_Fresh_20260922/CrossCarrier_Repro/EXPERIMENTSRESULT/REVISION_5090/SUPP_ABLATION/SCHED_fixedfull_ACR/seed42/checkpoints/best_pool_ema.pt`

### dev77 per-class F1
- Away: 0.0500
- Bend: 0.5500
- Kneel: 0.9000
- Pick: 0.8500
- SStep: 1.0000
- Sit: 0.2500
- Towards: 0.3500

### final77 per-class F1
- Away: 0.0750
- Bend: 0.7179
- Kneel: 0.9762
- Pick: 0.8750
- SStep: 1.0000
- Sit: 0.3514
- Towards: 0.2500
