# Public Baseline — Results (live progress doc)

_Authoritative spec: `PUBLIC_BASELINE_DESIGN.md` (v2). Method spec: `ABLATION_REPORT.md` §Method._
_Every number here is copied from actual tool output (`public_baseline/pb_results.json`)._
_Status: **ALL 11 MODELS DEPLOYED & RUN-CONFIRMED — awaiting "go" for full training.** No
formal 100ep×3seed training started; only dry-checks / 1-epoch smokes / eval-only reuse rows ran._

### Deployment readiness (2026-06-09)
| Rows | What was confirmed | Result |
|---|---|---|
| #1–6 generic | all 6 timm backbones load; 1-epoch pipeline smoke (convnext_t) | ✅ ready (params: vgg16_bn 134M, mobilenet 4.2M, effnet 4.0M, convnext 27.8M, swin 27.5M, convnextv2 27.9M) |
| #7 RadMamba | vendored + zeta shim + matplotlib; probe forward + 1-epoch smoke | ✅ ready (0.089M; **runs in-env, pure PyTorch**) |
| #8 SelaFD | vendored + ImageNet ViT-B loaded; probe forward + 1-epoch smoke | ✅ ready (14.34M trainable) |
| #9/#10/#11 reuse | unified eval of existing checkpoints | ✅ **final numbers done** (below), harness xcheck **0.0000** |

Fixed during deployment: pb_lib re-enables HF online (the v9 import chain force-sets
`HF_HUB_OFFLINE=1`, which blocked timm weight resolution); downloaded `vgg16_bn`; built
SelaFD pretrained `.pth`; vendored `zeta` pscan shim; installed `matplotlib` (torch/numpy
untouched). Only remaining step = launch the full sweep.

---

## 0. Protocol (identical for every row — the whole point)
| Item | Setting |
|---|---|
| Train bands | 10 / 24 GHz |
| Target | 77 GHz, fully held out (full 418-image test, 7 classes) |
| Epochs | 100 |
| Checkpoint | EMA weights at the FINAL epoch (no early stop, no target peek, no src-val gate) |
| Seeds | 42 / 1234 / 31415 → mean ± std |
| Input | 224×224×3 jet spectrogram, **per-image standardized** (== `tensor_transform(train=False)`); NOT ImageNet mean/std |
| Optimizer (generic #1–6) | AdamW lr 3e-4, wd 0.05, cosine + 3-ep warmup, batch 16, bf16, EMA 0.999 from ep5, class-balanced WeightedRandomSampler, label smoothing 0.05 |
| Primary metric | 77 GHz macro-F1 |
| Aux | 77 GHz acc, source-val macro-F1, gen-gap = src-val F1 − 77 GHz F1, trainable params, latency (ms/sample b1 bf16) |

**Augmentation policy (generic #1–6):** per-image standardization only — **no flip** (the
time axis encodes Away/Towards direction; flipping corrupts labels), **no DAS/HFT/SpecAugment**
(those are the method's contribution). This keeps "same training setup" = optimizer/schedule/
sampler/EMA, so any gap is method. _Flagged for confirmation — see §6._

---

## 1. Main table — SINGLE-model, head-to-head (#1–10)
_3-seed mean ± std. Filled by `pb_eval_unified.py` → `pb_results_auto.md`._

| # | Method | Family | 77GHz macro-F1 | 77GHz acc | src-val F1 | gen-gap | params (M) | latency (ms) | source |
|---|---|---|---|---|---|---|---|---|---|
| 1 | VGG16-BN | generic | _pending_ | | | | | | native |
| 2 | MobileNetV3-Large | generic | _pending_ | | | | | | native |
| 3 | EfficientNet-B0 | generic | _pending_ | | | | | | native |
| 4 | ConvNeXt-Tiny | generic | _pending_ | | | | | | native |
| 5 | Swin-Tiny | generic | _pending_ | | | | | | native |
| 6 | ConvNeXtV2-Tiny | generic | _pending_ | | | | | | native |
| 7 | RadMamba | radar SSM | _ready, pending train_ | | | | 0.09 | | external (runs in-env) |
| 8 | SelaFD-ViT-B/16 | radar PEFT | _ready, pending train_ | | | | 14.34 | | external |
| 9 | DINOv3-L + LoRA-only (no DAS/ACR) | foundation control | **0.302 ± 0.031** ✅ | 0.413 | **0.970** | **+0.668** | 1.85 | | reuse E1_noDAS |
| 10 | **Proposed — single (DAS+ACR)** | ours | **0.832 ± 0.034** ✅ | 0.836 ± 0.032 | 0.967 | +0.135 | 1.85 | ~25 | cite + reuse A_V13_GRL |

> Gap(#10 − #9) ≈ **+0.53** = the clean DAS+ACR cross-carrier contribution.
> Row 10 is recomputed via the unified evaluator (eval-only of the existing A_V13_GRL
> checkpoints) to fill src-val F1 / gen-gap / per-class; the headline 0.832 is the cite.

## 1b. Deployment row — SEPARATE, NOT head-to-head (#11)
| # | Method | 77GHz macro-F1 | 77GHz acc | boot95 CI | note |
|---|---|---|---|---|---|
| 11 | Proposed + 3-seed gap-aware ensemble | **0.856** ✅ (cite 0.857) | 0.859 | **[0.821, 0.889]** | params/latency ≈ 3×; do NOT compare to single-model #1–9 |

> ✅ rows #9/#10/#11 reproduced via the unified evaluator (eval-only of existing checkpoints):
> #9 0.3017±0.031, #10 0.8322±0.034 (per-seed 0.791/0.832/0.874), #11 0.8562/0.8589.
> Harness cross-check (278-subset vs history) **max|diff| = 0.0000**. Gap(#10−#9) = **+0.530**.

---

## 2. Row provenance (ran natively / reused / cited / un-runnable)
| # | Method | How obtained | Checkpoint location |
|---|---|---|---|
| 1–6 | generic backbones | **native** — train 100ep ×3 seeds (pb_train_generic.py) | `public_baseline/runs/<key>/seed<seed>/` |
| 7 | RadMamba | **external, env-risk** — attempt build; if mamba-ssm/causal-conv1d fail on Win+sm_120+torch2.11 → report un-run | `public_baseline/radmamba/` |
| 8 | SelaFD | **external** — locate repo+license, port LoRA/adapter recipe | `public_baseline/selafd/` |
| 9 | DINOv3-L LoRA-only | **reuse** — existing `E1_noDAS/seed{42,1234,31415}` final-EMA, eval-only | `REVISION_5090/E1_noDAS/` |
| 10 | Proposed single | **cite 0.832** + eval-only recompute of `A_V13_GRL` (= GRL/ACR w0.3) | `REVISION_5090/A_V13_GRL/` |
| 11 | Proposed ensemble | **cite 0.857** + bootstrap CI from the 3 A_V13_GRL seed posteriors | `REVISION_5090/A_V13_GRL/` |

---

## 3. External methods — repo / commit / license (VENDORED + RUN-CONFIRMED 2026-06-09)
| Method | Repo URL | Commit | License | Adaptations | Status |
|---|---|---|---|---|---|
| RadMamba | https://github.com/lab-emi/AIRHAR | `d49e2c7` | **Apache-2.0** | feed our 224×224×3 per-image-std input (channels=3); native recipe (AdamW lr5e-3, grad-clip 200, CE); vendored `zeta.nn.modules.p_scan.pscan` shim (== repo's own `selective_scan_seq`, no zetascale install); `dim=80,d_state=4,depth=1` (arguments.py defaults) | **DEPLOYED — probe+smoke OK, 0.089M params** |
| SelaFD | https://github.com/wangyijunlyy/SelaFD | `db67faf` | **⚠ NO LICENSE file in repo** (default = all-rights-reserved; academic reproduction/comparison only, no redistribution) | ViT-B/16 + LoRA(r4,α4) + serial/parallel Adapter; ImageNet `vit_base_patch16_224` loaded into vendored backbone (152/248 keys, adapters init); native recipe (Adam lr1e-3, Cosine, CE ls0.1); our 224×224×3 input | **DEPLOYED — probe+smoke OK, 14.34M trainable** |

> **RadMamba env-risk REVERSED (good news).** The design flagged RadMamba as Tier-3
> "likely won't compile (mamba-ssm/causal-conv1d CUDA kernels on Win+sm_120+torch2.11)".
> On inspection RadMamba's SSM is **pure PyTorch** (its own `selective_scan`/`selective_scan_seq`
> in `backbones/SSM.py`); it has **no mamba-ssm/causal-conv1d/triton dependency**. The only
> external symbol was `zeta.nn.modules.p_scan.pscan`, replaced by a 1-file numerically-identical
> shim (NOT installing zetascale, which pins torch and could downgrade Lider_5090). RadMamba
> therefore **runs natively in our environment** — it is NOT un-runnable. (Also installed the
> benign pure-python `matplotlib`, imported at module load by `RadMamba.py`; torch/numpy untouched.)

---

## 4. Generalization-gap story (confirm across the new baselines)
Expectation (from the old table): source-val saturates **0.97–0.99** while 77 GHz spreads
widely → gen-gap is large for weak cross-carrier methods, small for the method.
- [ ] confirm src-val F1 ≈ 0.97–0.99 for #1–6
- [ ] confirm 77 GHz F1 spreads (weak backbones low; #10 high)
- [ ] gap(#9) large (~0.6+), gap(#10) small

---

## 5. Supplement artifacts
- [ ] per-class P/R/F1 (Bend/Kneel/Pick/Sit collapse), worst-class F1 — `pb_results.json`
- [ ] confusion matrices
- [ ] per-seed raw macro-F1 / acc
- [ ] bootstrap 95% CI over the 418 target clips (every row + ensemble)
- [ ] trainable params per method; latency ms/sample (b1 bf16)
- [ ] training time / peak memory

---

## 6. Open design decisions (flagged — see chat)
1. **Generic-baseline augmentation** — **LOCKED (2026-06-09): per-image-std only, no flip,
   no DAS/HFT/SpecAug.** Cleanest "same training setup, method removed" baseline → the gap to
   #10 is attributable entirely to DAS+ACR. (Rejected: +flip corrupts Away/Towards direction;
   +HFT/SpecAug or +DAS would leak method into the baseline.)
2. **RadMamba scope** — **RESOLVED (2026-06-09): runs natively in-env.** RadMamba needs no
   mamba-ssm/causal-conv1d CUDA kernels (pure-PyTorch SSM + a 1-file `pscan` shim). Deployed,
   probe + 1-epoch smoke pass. No Linux box / no un-run fallback needed.

---

## 7. Sanity / cross-check
- Harness cross-check (proposed-family 278-subset final-EMA vs `history.json` last-epoch):
  **max|diff| = 0.0000** ✅ (re-run twice, identical) — harness validated.
- Any unstable training (NaN / divergence / seed collapse): none seen in smokes; watch the
  full sweep (esp. VGG16-BN full-FT on 644 imgs, and RadMamba lr5e-3).

---

## Runbook (after "go")
```bash
PY="C:/Users/Zirui Lin/anaconda3/envs/Lider_5090/python.exe"
cd "G:/zhanghe/Letter journal"
# 0) confirm all 6 timm backbones load in-env (no training):
"$PY" EXPERIMENTSRESULT/REVISION_5090/public_baseline/pb_run.py --dry-check
# 1) pipeline sanity (1 epoch):
"$PY" EXPERIMENTSRESULT/REVISION_5090/public_baseline/pb_run.py --smoke convnext_t
# 2) validate ONE method end-to-end (single seed) then scale:
"$PY" EXPERIMENTSRESULT/REVISION_5090/public_baseline/pb_run.py --backbones convnext_t --seeds 42
# 3) full sweep (#1-6 x 3 seeds), background + log:
"$PY" EXPERIMENTSRESULT/REVISION_5090/public_baseline/pb_run.py   # > sweep.log 2>&1 &
# 3b) external methods (#7 RadMamba, #8 SelaFD) x 3 seeds (deployed + smoke-OK):
for s in 42 1234 31415; do
  "$PY" EXPERIMENTSRESULT/REVISION_5090/public_baseline/pb_external.py --train radmamba --seed $s
  "$PY" EXPERIMENTSRESULT/REVISION_5090/public_baseline/pb_external.py --train selafd   --seed $s
done
# 4) unified eval (fills table; #1-8 native, #9/#10 reuse, #11 ensemble, cross-check, CIs):
"$PY" EXPERIMENTSRESULT/REVISION_5090/public_baseline/pb_eval_unified.py --latency
```
The unified evaluator REGISTRY already routes #7→`runs/radmamba`, #8→`runs/selafd`
(external family branch wired + regression-checked); they auto-populate once trained.
Run via the **Bash tool**, env vars exported, redirect to a log, `run_in_background=true`
(a PowerShell pipe with `2>&1` + ErrorAction Stop kills training — tqdm writes to stderr).
