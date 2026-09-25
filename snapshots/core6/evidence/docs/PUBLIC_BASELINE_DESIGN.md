# Public Baseline — Design Report (v2)
_Supersedes the old generic-backbone baseline table (`baselines/README.md`, 13
ImageNet backbones, 80 ep, 77 GHz-dev selection). That table is deprecated; its
CODE/infra is reused, its RESULTS are not._

## 1. Why redesign
The old public baseline was **13 generic vision backbones** (CLIP / Swin / MobileNet /
ConvNeXt / EfficientNet-B0/B3 / EVA / ConvNeXtV2 / DINOv3-B …), all full fine-tuned,
**80 epochs**, **selected by 77 GHz-dev macro-F1**. Three problems a reviewer will hit:
1. **"You only compared against unrelated vision models."** No radar-specific method, no
   PEFT method, no legacy radar-transfer baseline.
2. **Target leakage in selection.** 77 GHz-dev was used to pick checkpoints — the method
   uses final-EMA (no peek), so the comparison is both inconsistent and ironically
   *advantages* the baselines.
3. **Protocol mismatch** (80 ep / dev-selection / old env) vs the method (100 ep /
   final-EMA / Lider_5090).

The redesign covers **weak→strong AND every reviewer-relevant family**: legacy
radar-transfer CNN, lightweight CNN, efficient CNN, modern CNN, hierarchical
transformer, modern-CNN/SSL, **radar-specific SSM**, **radar PEFT ViT**, and a
**controlled foundation+LoRA** baseline that isolates our contribution.

## 2. Protocol (identical for every method — put this table in the paper)
| Item | Setting |
|---|---|
| Train bands | 10 / 24 GHz only |
| Target band | 77 GHz, fully held out |
| Epochs | **100** |
| Checkpoint | **EMA weights at the final epoch** (no early stopping) |
| Selection | **No 77 GHz and no source-val used** for training, tuning, or checkpoint selection |
| Seeds | **3** (42 / 1234 / 31415); report mean ± std |
| Input (image backbones) | same 224×224×3 jet-rendered spectrogram, per-image standardized |
| Input (radar-native models) | same STFT crop, adapted to the method's native channel format |
| Primary metric | **77 GHz macro-F1** |
| Aux metrics | 77 GHz accuracy, source-val macro-F1, generalization gap, trainable params, latency |

**Rationale for final-EMA (write this).** In the old table multiple backbones reached
**source-val macro-F1 0.97–0.99 yet wildly different 77 GHz macro-F1** → source-val is
saturated and useless as a selection signal. So no source-val selection; report the
final-epoch EMA checkpoint for all methods. Eliminates the "how did you select
checkpoints?" question.

## 3. Main table (10 rows)
| # | Method | Family | FT / training mode | Why it belongs |
|---|---|---|---|---|
| 1 | **VGG16-BN** | Legacy CNN / radar transfer | ImageNet pretrained, **full FT** | The historical radar micro-Doppler transfer baseline (CI4R-era ImageNet-VGG16). Most domain history. |
| 2 | **MobileNetV3-Large** | Lightweight CNN | ImageNet, full FT | "Is a small mobile CNN enough?" |
| 3 | **EfficientNet-B0** | Efficient CNN | ImageNet, full FT | Compound-scaled efficient transfer baseline. |
| 4 | **ConvNeXt-Tiny** | Modern CNN | ImageNet, full FT | Strong ConvNet inductive bias — can a modern CNN solve cross-carrier? |
| 5 | **Swin-Tiny** | Hierarchical Transformer | ImageNet, full FT | Standard hierarchical ViT backbone. |
| 6 | **ConvNeXtV2-Tiny** | Modern CNN + SSL/MAE | public pretrained, full FT | Stronger modern CNN (FCMAE+GRN). |
| 7 | **RadMamba** | Radar micro-Doppler SSM | native / official recipe | Radar-specific public method (Mamba SSM for micro-Doppler HAR). Answers "no radar baseline?". |
| 8 | **SelaFD-ViT-B/16** | Radar ViT + LoRA/adapter | official LoRA/adapter recipe | Most-relevant radar PEFT baseline (ViT + LoRA on Time-Doppler signatures). |
| 9 | **DINOv3 ViT-L/16 + LoRA-only** | Foundation control | frozen DINOv3 + same LoRA/head, **no DAS / no ACR** | The key controlled baseline: same backbone + LoRA as ours, minus the cross-carrier modules. |
| 10 | **Proposed — single model** | Ours | DINOv3 + LoRA + **DAS + ACR** | **Primary head-to-head row** (single-model, 3-seed mean±std). Gap to #9 = clean cross-carrier contribution. |
| 11 | **Proposed + 3-seed gap-aware ensemble** | Ours (deployment) | averages the 3 single-model seeds' posteriors | Deployment add-on, **reported separately** — do NOT pit against single-model baselines (you'd have to ensemble them too). Single ensemble → no seed-std (use bootstrap CI). |

> **⚠️ Fairness — single vs ensemble.** The main comparison is **single-model vs
> single-model** (row 10 vs rows 1–9, all 3-seed mean±std). The 3-seed ensemble (row 11)
> is OUR deployment step; present it as a separate row, not as the head-to-head number,
> or you invite "you only ensembled your own method." Latency/params for row 11 are 3× the
> single model.

> **Correction vs the original brief (it targeted the OLD system).** The current Proposed
> is **DAS + ACR (adversarial carrier-residual / GRL)** — NOT "residual split +
> falsification". The falsification (V15) module was **ablated out and rejected**
> (V15R_RESULTS.md, 0.823 < 0.832), and the old covariance "residual split" was **replaced
> by the ACR GRL head**. Verified strongest config: **single-model 0.832 ± 0.034**
> (row 10), **ensemble 0.857 / acc 0.859** (row 11). The gap to #9 estimates **DAS + ACR**,
> not "physics + falsification".

> **#9 already exists.** Frozen DINOv3-L + LoRA + recipe with DAS/ACR OFF = the existing
> `E1_noDAS` run (≈ **0.302** macro-F1, 3 seeds on disk). Reuse those checkpoints; only
> re-evaluate under final-EMA / full-418. Gap(#10 − #9) ≈ +0.53 = the whole cross-carrier
> contribution.

## 4. Metrics
**Main table (paper body):**
| Metric | Why |
|---|---|
| 77 GHz macro-F1 (mean±std) | primary; robust to class imbalance / confusable activities |
| 77 GHz accuracy | comparability with radar-HAR literature |
| source-val macro-F1 | shows source saturation |
| **generalization gap = src-val F1 − 77 GHz F1** | shows cross-carrier collapse directly |
| trainable params | **compute per method**; ours ≈ 2.38 M *deployed* (the ACR adversary head is training-only, discarded at inference) vs full-FT baselines' tens–hundreds of M |
| latency (ms/sample) | engineering credibility; ours ≈ 25 ms/sample **single model** (the 3-seed ensemble row is ≈3× ≈ 75 ms) |

**Supplement:** per-class P/R/F1 (Bend/Kneel/Pick/Sit collapse), worst-class F1,
confusion matrix, **per-seed raw results**, **bootstrap 95 % CI over the 418 target
clips** (seed-std only reflects training randomness, not test-set sampling),
training time / peak memory, optional FLOPs/MACs.

## 5. Fine-tuning policy (do NOT force-uniform)
- **Generic image backbones (#1–6):** full fine-tune (so you can't be accused of
  under-training them).
- **Radar-specific / PEFT (#7 RadMamba, #8 SelaFD):** their **native** recipe (don't
  mutate them into "ours with a different backbone").
- **#9 DINOv3-LoRA-only:** LoRA-only, same rank/head as ours, no cross-carrier modules.
- Paper sentence: _"Public CNN/Transformer baselines are fully fine-tuned to avoid
  under-training them. Radar-specific and parameter-efficient baselines follow their
  native training recipes. The DINOv3-LoRA baseline isolates the contribution of the
  proposed Doppler-axis resampling and adversarial carrier-residual losses."_

## 6. Demoted to supplement / dropped
| Method | Action | Reason |
|---|---|---|
| EfficientNet-B3 | drop (keep B0) | redundant with B0 |
| CLIP ViT-L/16 | supplement | text-image pretraining weakly related to radar physics |
| EVA-02 ViT-L/16 | supplement | redundant with DINOv3 as the strong-ViT control |
| ResNet-18/50 | optional alt to VGG16 | keep VGG16 (CI4R history); ResNet-50 only if VGG16 deemed too old |

## 7. Implementation risk tiers (read before scoping)
- **Tier 1 — easy (reuse existing pipeline), #1–6 + #9:** timm backbones via the
  `baselines/` infra (data loaders, timm wrapper) retargeted to the new protocol
  (100 ep, final-EMA, full-418); #9 = reuse `E1_noDAS` checkpoints + re-eval.
- **Tier 2 — real engineering, #8 SelaFD:** find the official repo + license, port their
  LoRA/adapter recipe, adapt their dataloader to our 7-class 10/24→77 split.
- **Tier 3 — external + ENVIRONMENT RISK, #7 RadMamba:** Mamba SSM needs custom CUDA
  kernels (`mamba-ssm` / `causal-conv1d`). On **Windows + RTX 5090 (sm_120) + torch 2.11**
  these very likely **do not compile** (historically Linux-only; sm_120 is brand new).
  Plan: try to build; if it fails, run on a Linux box, OR report it as
  **"not reproducible in our environment"** — **never** fabricate a RadMamba number.

## 8. Hard rules for whoever runs this
- **NO fabricated metrics.** Every number comes from actual tool output. An external
  method that won't run is reported as un-run, not invented.
- **Same harness.** Reuse `baseline_v20/aggregate_ablation_finalema.py`'s eval (full-418,
  final-EMA, history cross-check must stay ~0.0000).
- **License check** before vendoring any external repo (RadMamba/SelaFD); record source +
  commit + license in the results file.
- **Protocol parity** is the whole point: same epochs, same final-EMA, same no-target-
  selection, same 224×224 input for the image backbones.
