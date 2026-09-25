# CrossCarrier reproducibility backup — 2026-09-25

## Original7

Source snapshot:
Original7_Code/

Original working directory:
/scratch/st-zliu-1/heqingz/CrossCarrier_Fresh_20260922/CrossCarrier_Repro

Task:
7-class CrossCarrier HAR.
Primary historical direction:
10+24 GHz -> 77 GHz.

Fresh extensions include:
- three held-out-frequency grouped protocols,
- proposed vs no_das_acr,
- supplementary 7 x 3 ablations,
- fresh anchor training,
- exact final-EMA evaluation scripts.

Classes:
Away, Bend, Kneel, Pick, SStep, Sit, Towards.

Seeds:
42, 1234, 31415.

Training:
100 epochs.
Final epoch EMA.
V100 FP16 for fresh Sockeye runs.

Important:
Raw radar data, DINOv3 backbone weights, and training checkpoints are NOT
included in this source archive.


## CoreMovements6

Source snapshot:
Core6_Code/

Original working directory:
/scratch/st-zliu-1/heqingz/CrossCarrier_Core6_FullSuite_20260924

Classes:
Away, Bend, Kneel, Pick, Sit, Towards.

Domains:
10 GHz, 24 GHz, 77 GHz.

Recorded held-out-frequency folds:
24+77 -> 10
10+77 -> 24
10+24 -> 77

Formal full suite:
10 recipes x 3 target frequencies x 3 seeds = 90 runs.

Seeds:
42, 1234, 31415.

Training:
100 epochs.
Final epoch EMA.
Target-blind model selection.
V100 FP16.

The original CoreMovements6 image partitions are preserved.
Documented duplicate-risk groups are not silently removed.

Raw image data, DINOv3 backbone weights, active batch outputs, and model
checkpoints are NOT included in this source archive.


## Current experiment provenance

Original7 exact-anchor re-evaluation:
Job 13084058

Core6 full 90-run array:
Job 13062778

Earlier Original7 fresh supplementary ablation:
Job 13036603

Earlier grouped proposed/no_das_acr experiment:
Job 13036553


## GitHub

Repository:
HeqingNRC/CrossCarrier-Repro-heqing

Backup branch:
backup-repro-20260925

Large model/data artifacts are intentionally excluded from GitHub.
