# Pool selection summary

- threshold: source_val_acc_ema >= 0.90
- period: every 10 epochs
- pool size: 5
- selection metric: dev77_f1

## Members

| epoch | src_val_acc | src_val_F1 | dev77 acc | dev77 F1 | final77 acc | final77 F1 |
|---:|---:|---:|---:|---:|---:|---:|
|  60 | 0.9259 | 0.9258 | 0.1714 | 0.0735 | 0.2230 | 0.1224 |
|  70 | 0.9568 | 0.9577 | 0.2000 | 0.0991 | 0.2482 | 0.1284 |
|  80 | 0.9691 | 0.9695 | 0.2214 | 0.1162 | 0.2554 | 0.1288 |
|  90 | 0.9753 | 0.9757 | 0.2500 | 0.1259 | 0.2626 | 0.1300 |
| 100 | 0.9815 | 0.9818 | 0.2571 | 0.1277 | 0.2662 | 0.1289 |

## Selected (best in pool)

- epoch: **100**
- source val acc / F1 (EMA): 0.9815 / 0.9818
- 77GHz dev   acc / F1: 0.2571 / 0.1277
- 77GHz final acc / F1: 0.2662 / 0.1289
- checkpoint: `/scratch/st-zliu-1/heqingz/CrossCarrier_Fresh_20260922/CrossCarrier_Repro/EXPERIMENTSRESULT/REVISION_5090/SUPP_ABLATION/DAS_ERM/seed42/checkpoints/best_pool_ema.pt`

### dev77 per-class F1
- Away: 0.8000
- Bend: 1.0000
- Kneel: 0.0000
- Pick: 0.0000
- SStep: 0.0000
- Sit: 0.0000
- Towards: 0.0000

### final77 per-class F1
- Away: 0.8750
- Bend: 1.0000
- Kneel: 0.0000
- Pick: 0.0000
- SStep: 0.0000
- Sit: 0.0000
- Towards: 0.0000
