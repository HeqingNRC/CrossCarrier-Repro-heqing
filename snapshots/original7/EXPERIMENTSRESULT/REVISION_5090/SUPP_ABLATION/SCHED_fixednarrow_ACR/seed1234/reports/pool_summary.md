# Pool selection summary

- threshold: source_val_acc_ema >= 0.90
- period: every 10 epochs
- pool size: 7
- selection metric: dev77_f1

## Members

| epoch | src_val_acc | src_val_F1 | dev77 acc | dev77 F1 | final77 acc | final77 F1 |
|---:|---:|---:|---:|---:|---:|---:|
|  40 | 0.9321 | 0.9343 | 0.3500 | 0.3162 | 0.4173 | 0.3865 |
|  50 | 0.9753 | 0.9757 | 0.7143 | 0.7131 | 0.7050 | 0.6939 |
|  60 | 0.9691 | 0.9688 | 0.7929 | 0.7917 | 0.8381 | 0.8386 |
|  70 | 0.9630 | 0.9627 | 0.8071 | 0.8098 | 0.8633 | 0.8664 |
|  80 | 0.9691 | 0.9687 | 0.8214 | 0.8204 | 0.8957 | 0.8961 |
|  90 | 0.9630 | 0.9625 | 0.8286 | 0.8256 | 0.8993 | 0.8998 |
| 100 | 0.9630 | 0.9625 | 0.8429 | 0.8397 | 0.8885 | 0.8888 |

## Selected (best in pool)

- epoch: **100**
- source val acc / F1 (EMA): 0.9630 / 0.9625
- 77GHz dev   acc / F1: 0.8429 / 0.8397
- 77GHz final acc / F1: 0.8885 / 0.8888
- checkpoint: `/scratch/st-zliu-1/heqingz/CrossCarrier_Fresh_20260922/CrossCarrier_Repro/EXPERIMENTSRESULT/REVISION_5090/SUPP_ABLATION/SCHED_fixednarrow_ACR/seed1234/checkpoints/best_pool_ema.pt`

### dev77 per-class F1
- Away: 1.0000
- Bend: 0.5500
- Kneel: 0.9000
- Pick: 0.8500
- SStep: 1.0000
- Sit: 0.7000
- Towards: 0.9000

### final77 per-class F1
- Away: 1.0000
- Bend: 0.7436
- Kneel: 0.8810
- Pick: 0.8750
- SStep: 1.0000
- Sit: 0.7838
- Towards: 0.9250
