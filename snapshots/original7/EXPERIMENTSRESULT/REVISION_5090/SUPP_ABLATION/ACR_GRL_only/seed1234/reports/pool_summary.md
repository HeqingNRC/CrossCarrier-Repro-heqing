# Pool selection summary

- threshold: source_val_acc_ema >= 0.90
- period: every 10 epochs
- pool size: 7
- selection metric: dev77_f1

## Members

| epoch | src_val_acc | src_val_F1 | dev77 acc | dev77 F1 | final77 acc | final77 F1 |
|---:|---:|---:|---:|---:|---:|---:|
|  40 | 0.9691 | 0.9695 | 0.4071 | 0.3425 | 0.4460 | 0.3776 |
|  50 | 0.9691 | 0.9695 | 0.6500 | 0.6141 | 0.6978 | 0.6578 |
|  60 | 0.9691 | 0.9694 | 0.7643 | 0.7606 | 0.8345 | 0.8329 |
|  70 | 0.9691 | 0.9694 | 0.8071 | 0.8090 | 0.8705 | 0.8694 |
|  80 | 0.9691 | 0.9694 | 0.8143 | 0.8145 | 0.8669 | 0.8654 |
|  90 | 0.9753 | 0.9755 | 0.8143 | 0.8135 | 0.8669 | 0.8644 |
| 100 | 0.9753 | 0.9755 | 0.8214 | 0.8200 | 0.8705 | 0.8684 |

## Selected (best in pool)

- epoch: **100**
- source val acc / F1 (EMA): 0.9753 / 0.9755
- 77GHz dev   acc / F1: 0.8214 / 0.8200
- 77GHz final acc / F1: 0.8705 / 0.8684
- checkpoint: `/scratch/st-zliu-1/heqingz/CrossCarrier_Fresh_20260922/CrossCarrier_Repro/EXPERIMENTSRESULT/REVISION_5090/SUPP_ABLATION/ACR_GRL_only/seed1234/checkpoints/best_pool_ema.pt`

### dev77 per-class F1
- Away: 1.0000
- Bend: 0.7000
- Kneel: 0.7500
- Pick: 0.8500
- SStep: 0.9500
- Sit: 0.6500
- Towards: 0.8500

### final77 per-class F1
- Away: 1.0000
- Bend: 0.7949
- Kneel: 0.8333
- Pick: 0.8500
- SStep: 1.0000
- Sit: 0.7838
- Towards: 0.8250
