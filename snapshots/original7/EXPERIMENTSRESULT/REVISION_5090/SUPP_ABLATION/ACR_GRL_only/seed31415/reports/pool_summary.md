# Pool selection summary

- threshold: source_val_acc_ema >= 0.90
- period: every 10 epochs
- pool size: 8
- selection metric: dev77_f1

## Members

| epoch | src_val_acc | src_val_F1 | dev77 acc | dev77 F1 | final77 acc | final77 F1 |
|---:|---:|---:|---:|---:|---:|---:|
|  30 | 0.9012 | 0.9035 | 0.3000 | 0.2241 | 0.3022 | 0.2138 |
|  40 | 0.9444 | 0.9451 | 0.5000 | 0.4160 | 0.5036 | 0.4412 |
|  50 | 0.9568 | 0.9576 | 0.6429 | 0.6139 | 0.7194 | 0.7093 |
|  60 | 0.9630 | 0.9634 | 0.7643 | 0.7623 | 0.8237 | 0.8222 |
|  70 | 0.9691 | 0.9694 | 0.8143 | 0.8127 | 0.8669 | 0.8658 |
|  80 | 0.9691 | 0.9694 | 0.8214 | 0.8188 | 0.8597 | 0.8571 |
|  90 | 0.9691 | 0.9694 | 0.8214 | 0.8184 | 0.8669 | 0.8656 |
| 100 | 0.9691 | 0.9694 | 0.8071 | 0.8020 | 0.8597 | 0.8579 |

## Selected (best in pool)

- epoch: **80**
- source val acc / F1 (EMA): 0.9691 / 0.9694
- 77GHz dev   acc / F1: 0.8214 / 0.8188
- 77GHz final acc / F1: 0.8597 / 0.8571
- checkpoint: `/scratch/st-zliu-1/heqingz/CrossCarrier_Fresh_20260922/CrossCarrier_Repro/EXPERIMENTSRESULT/REVISION_5090/SUPP_ABLATION/ACR_GRL_only/seed31415/checkpoints/best_pool_ema.pt`

### dev77 per-class F1
- Away: 0.9500
- Bend: 0.6500
- Kneel: 0.9000
- Pick: 0.8000
- SStep: 1.0000
- Sit: 0.6000
- Towards: 0.8500

### final77 per-class F1
- Away: 1.0000
- Bend: 0.7692
- Kneel: 0.8571
- Pick: 0.8250
- SStep: 1.0000
- Sit: 0.6486
- Towards: 0.9000
