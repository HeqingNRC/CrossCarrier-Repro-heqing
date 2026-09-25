# Pool selection summary

- threshold: source_val_acc_ema >= 0.90
- period: every 10 epochs
- pool size: 7
- selection metric: dev77_f1

## Members

| epoch | src_val_acc | src_val_F1 | dev77 acc | dev77 F1 | final77 acc | final77 F1 |
|---:|---:|---:|---:|---:|---:|---:|
|  40 | 0.9074 | 0.9095 | 0.2214 | 0.0988 | 0.2230 | 0.0993 |
|  50 | 0.9321 | 0.9324 | 0.2143 | 0.1080 | 0.2050 | 0.0992 |
|  60 | 0.9444 | 0.9453 | 0.1571 | 0.0643 | 0.1655 | 0.0690 |
|  70 | 0.9444 | 0.9453 | 0.1500 | 0.0596 | 0.1583 | 0.0612 |
|  80 | 0.9383 | 0.9392 | 0.1500 | 0.0596 | 0.1475 | 0.0430 |
|  90 | 0.9444 | 0.9449 | 0.1429 | 0.0480 | 0.1475 | 0.0431 |
| 100 | 0.9321 | 0.9326 | 0.1429 | 0.0480 | 0.1475 | 0.0431 |

## Selected (best in pool)

- epoch: **50**
- source val acc / F1 (EMA): 0.9321 / 0.9324
- 77GHz dev   acc / F1: 0.2143 / 0.1080
- 77GHz final acc / F1: 0.2050 / 0.0992
- checkpoint: `/scratch/st-zliu-1/heqingz/CrossCarrier_Fresh_20260922/CrossCarrier_Repro/EXPERIMENTSRESULT/REVISION_5090/SUPP_ABLATION/DAS_ERM/seed31415/checkpoints/best_pool_ema.pt`

### dev77 per-class F1
- Away: 0.0000
- Bend: 0.5500
- Kneel: 0.0000
- Pick: 0.0000
- SStep: 0.0000
- Sit: 0.0000
- Towards: 0.9500

### final77 per-class F1
- Away: 0.0000
- Bend: 0.4359
- Kneel: 0.0000
- Pick: 0.0000
- SStep: 0.0000
- Sit: 0.0000
- Towards: 1.0000
