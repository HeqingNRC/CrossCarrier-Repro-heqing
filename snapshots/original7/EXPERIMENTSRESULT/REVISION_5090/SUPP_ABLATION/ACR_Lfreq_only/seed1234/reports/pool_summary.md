# Pool selection summary

- threshold: source_val_acc_ema >= 0.90
- period: every 10 epochs
- pool size: 7
- selection metric: dev77_f1

## Members

| epoch | src_val_acc | src_val_F1 | dev77 acc | dev77 F1 | final77 acc | final77 F1 |
|---:|---:|---:|---:|---:|---:|---:|
|  40 | 0.9444 | 0.9450 | 0.4714 | 0.4087 | 0.4964 | 0.4392 |
|  50 | 0.9877 | 0.9878 | 0.7500 | 0.7424 | 0.7698 | 0.7547 |
|  60 | 0.9753 | 0.9757 | 0.8357 | 0.8336 | 0.8741 | 0.8715 |
|  70 | 0.9753 | 0.9755 | 0.8500 | 0.8459 | 0.8705 | 0.8670 |
|  80 | 0.9691 | 0.9692 | 0.8286 | 0.8240 | 0.8669 | 0.8649 |
|  90 | 0.9691 | 0.9692 | 0.8286 | 0.8259 | 0.8633 | 0.8615 |
| 100 | 0.9691 | 0.9692 | 0.8214 | 0.8188 | 0.8669 | 0.8657 |

## Selected (best in pool)

- epoch: **70**
- source val acc / F1 (EMA): 0.9753 / 0.9755
- 77GHz dev   acc / F1: 0.8500 / 0.8459
- 77GHz final acc / F1: 0.8705 / 0.8670
- checkpoint: `/scratch/st-zliu-1/heqingz/CrossCarrier_Fresh_20260922/CrossCarrier_Repro/EXPERIMENTSRESULT/REVISION_5090/SUPP_ABLATION/ACR_Lfreq_only/seed1234/checkpoints/best_pool_ema.pt`

### dev77 per-class F1
- Away: 0.9500
- Bend: 0.8000
- Kneel: 0.8500
- Pick: 0.8000
- SStep: 1.0000
- Sit: 0.6000
- Towards: 0.9500

### final77 per-class F1
- Away: 1.0000
- Bend: 0.8974
- Kneel: 0.8571
- Pick: 0.7750
- SStep: 1.0000
- Sit: 0.6486
- Towards: 0.9000
