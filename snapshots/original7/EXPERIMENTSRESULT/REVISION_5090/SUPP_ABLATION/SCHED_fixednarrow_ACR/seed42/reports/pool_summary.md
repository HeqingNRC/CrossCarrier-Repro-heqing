# Pool selection summary

- threshold: source_val_acc_ema >= 0.90
- period: every 10 epochs
- pool size: 7
- selection metric: dev77_f1

## Members

| epoch | src_val_acc | src_val_F1 | dev77 acc | dev77 F1 | final77 acc | final77 F1 |
|---:|---:|---:|---:|---:|---:|---:|
|  40 | 0.9012 | 0.9004 | 0.4643 | 0.4198 | 0.5036 | 0.4474 |
|  50 | 0.9691 | 0.9695 | 0.6857 | 0.6288 | 0.7122 | 0.6563 |
|  60 | 0.9753 | 0.9757 | 0.7214 | 0.6840 | 0.7698 | 0.7341 |
|  70 | 0.9753 | 0.9757 | 0.7929 | 0.7701 | 0.8201 | 0.7803 |
|  80 | 0.9815 | 0.9817 | 0.7857 | 0.7626 | 0.8417 | 0.8137 |
|  90 | 0.9815 | 0.9817 | 0.8071 | 0.7884 | 0.8597 | 0.8412 |
| 100 | 0.9753 | 0.9755 | 0.7929 | 0.7735 | 0.8489 | 0.8303 |

## Selected (best in pool)

- epoch: **90**
- source val acc / F1 (EMA): 0.9815 / 0.9817
- 77GHz dev   acc / F1: 0.8071 / 0.7884
- 77GHz final acc / F1: 0.8597 / 0.8412
- checkpoint: `/scratch/st-zliu-1/heqingz/CrossCarrier_Fresh_20260922/CrossCarrier_Repro/EXPERIMENTSRESULT/REVISION_5090/SUPP_ABLATION/SCHED_fixednarrow_ACR/seed42/checkpoints/best_pool_ema.pt`

### dev77 per-class F1
- Away: 1.0000
- Bend: 0.7500
- Kneel: 0.9500
- Pick: 0.7500
- SStep: 1.0000
- Sit: 0.3000
- Towards: 0.9000

### final77 per-class F1
- Away: 1.0000
- Bend: 0.7949
- Kneel: 1.0000
- Pick: 0.9000
- SStep: 1.0000
- Sit: 0.3514
- Towards: 0.9250
