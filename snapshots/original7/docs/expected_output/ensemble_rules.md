# P0-C ensemble fairness (PRELIMINARY -- eval-only, existing checkpoints)

Full-418 77GHz, final-EMA, seeds 42/1234/31415. Single = per-seed mean+/-std.

| Config | Single macro-F1 | Ens majority | Ens logit-avg | Ens posterior |
|---|---|---|---|---|
| DAS+ACR (A_V13_GRL) | 0.8322 ± 0.0337 | 0.8512 | 0.8566 | 0.8562 |

_Accuracy and per-seed values in ensemble_rules.json. Logits cached in logits_cache.npz for paired_bootstrap_ci.py._