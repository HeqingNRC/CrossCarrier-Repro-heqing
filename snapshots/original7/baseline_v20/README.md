# V20: V15 + V13

V20 combines the V15 DAS-falsification idea with the V13 carrier-residual representation split. It is built directly from the `baseline_v9_2_1` no-pair baseline, not from V11-V14 or any other intermediate branch.

## Shared Baseline Framework

V20 keeps the same baseline framework except for the two intended method additions:

- DINOv3 ViT-L/16 backbone
- LoRA rank 2 on `qkv`, `proj`, `fc1`, `fc2`
- ArcFace + LogitAdjust
- SupCon
- MIRO frozen-oracle alignment
- DAS curriculum, HFT, Radar SpecAugment
- EMA and class-balanced source sampler
- no 10/24GHz pair-consistency loss
- 100 training epochs in current experiments

## Protocol

Paper-facing results should be interpreted under the strict protocol:

```text
Choose the checkpoint with the best source-domain validation performance,
then report that checkpoint's 77GHz generalization performance.
```

## Current Selected-Seed Result

The selected-seed V20 table is recorded at:

```text
G:\zhanghe\EXPERIMENTSRESULT\V20_3seed_source90\README.md
```

That file currently tracks the seeds the author wants to focus on for follow-up comparisons.
