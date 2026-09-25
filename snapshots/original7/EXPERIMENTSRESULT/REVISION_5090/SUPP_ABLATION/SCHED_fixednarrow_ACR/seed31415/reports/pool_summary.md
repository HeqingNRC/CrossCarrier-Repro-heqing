# Pool selection summary

- threshold: source_val_acc_ema >= 0.90
- period: every 10 epochs
- pool size: 7
- selection metric: dev77_f1

## Members

| epoch | src_val_acc | src_val_F1 | dev77 acc | dev77 F1 | final77 acc | final77 F1 |
|---:|---:|---:|---:|---:|---:|---:|
|  40 | 0.9259 | 0.9280 | 0.4571 | 0.3762 | 0.4604 | 0.3824 |
|  50 | 0.9630 | 0.9642 | 0.6071 | 0.5696 | 0.6439 | 0.5854 |
|  60 | 0.9877 | 0.9878 | 0.6929 | 0.6654 | 0.7482 | 0.7185 |
|  70 | 0.9877 | 0.9878 | 0.7786 | 0.7646 | 0.8058 | 0.7795 |
|  80 | 0.9877 | 0.9878 | 0.8000 | 0.7894 | 0.8345 | 0.8223 |
|  90 | 0.9877 | 0.9878 | 0.8000 | 0.7911 | 0.8453 | 0.8353 |
| 100 | 0.9877 | 0.9878 | 0.8071 | 0.7949 | 0.8417 | 0.8320 |

## Selected (best in pool)

- epoch: **100**
- source val acc / F1 (EMA): 0.9877 / 0.9878
- 77GHz dev   acc / F1: 0.8071 / 0.7949
- 77GHz final acc / F1: 0.8417 / 0.8320
- checkpoint: `/scratch/st-zliu-1/heqingz/CrossCarrier_Fresh_20260922/CrossCarrier_Repro/EXPERIMENTSRESULT/REVISION_5090/SUPP_ABLATION/SCHED_fixednarrow_ACR/seed31415/checkpoints/best_pool_ema.pt`

### dev77 per-class F1
- Away: 1.0000
- Bend: 0.7000
- Kneel: 1.0000
- Pick: 0.5500
- SStep: 1.0000
- Sit: 0.4500
- Towards: 0.9500

### final77 per-class F1
- Away: 1.0000
- Bend: 0.7436
- Kneel: 0.9286
- Pick: 0.7250
- SStep: 1.0000
- Sit: 0.4865
- Towards: 0.9750
