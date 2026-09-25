# Known People, Unknown Frequency

Task-specific reconstruction for the current single-dataset setting:
Alabama data plus the author's added 24GHz samples.

## Goal

- Train/val on known people using only 10GHz + 24GHz.
- Test on the unseen frequency 77GHz.
- Every test subject is already seen in source-frequency training data.
- Validation stays within the same known subjects via image-level split.
- Current paper reporting uses the strict protocol: select the source-domain
  best checkpoint and report its 77GHz generalization result.

## Known subjects

- ua_0000, ua_0010, ua_0014, ua_0029, ua_0030

## Split policy

- Source frequencies: 10GHz, 24GHz
- Target frequency: 77GHz
- Validation fraction per (frequency, subject, class): 20%
- Groups smaller than 5 images stay entirely in train.
- RNG seed: 42

## Counts

- train: 1024
- val: 253
- test: 653

### By split x frequency

- test|77GHz: 653
- train|10GHz: 481
- train|24GHz: 543
- val|10GHz: 121
- val|24GHz: 132

## Notes

- Original dataset and original manifests were left untouched.
- Manifest paths are repo-relative so downstream configs can point to this task manifest directory.
- Summary JSON: `tasks/known_people_unknown_freq/summary.json`

## Incomplete subject-class-frequency coverage

These subject/class pairs are not present in all three frequencies.
That is expected from the raw capture set and is recorded here for transparency.

- ua_0000 | Away | present: 24GHz | missing: 10GHz, 77GHz
- ua_0000 | Towards | present: 24GHz, 77GHz | missing: 10GHz
- ua_0010 | Kneel | present: 24GHz, 77GHz | missing: 10GHz
- ua_0010 | Sit | present: 24GHz, 77GHz | missing: 10GHz
- ua_0014 | Towards | present: 24GHz | missing: 10GHz, 77GHz
