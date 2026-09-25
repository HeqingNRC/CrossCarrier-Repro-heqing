> 本次三方向扩展与已实测结果：先阅读 [CROSS_FREQUENCY_GUIDE_zh.md](CROSS_FREQUENCY_GUIDE_zh.md)；Sockeye 操作见 [hpc/README_zh.md](hpc/README_zh.md)。下方为原复现包说明，旧重训命令及默认参数请以新指南为准。

# Zero-Shot Cross-Carrier Transfer for Radar Micro-Doppler HAR — 复现包

论文：`paper/Zero-Shot Cross-Carrier Transfer for Radar.pdf`（IEEE Signal Processing Letters 投稿版）

这个包里的东西**足够复现论文的最强系统**：训练用 10 + 24 GHz 雷达微多普勒时频图，**零样本（zero-shot）测试在完全没见过的 77 GHz 频段**上。跑一条命令即可，不需要重新训练。

---

## 1. 任务和方法一句话说明

| | |
|---|---|
| **任务** | 7 类人体活动识别（Away / Bend / Kneel / Pick / SStep / Sit / Towards） |
| **训练频段** | 10 GHz + 24 GHz |
| **测试频段** | **77 GHz，完全留出**，418 张图全测 |
| **泛化维度** | **只跨载频（cross-carrier）**。训练和测试是**同一批人**——不是跨人（不要写成 unseen-subject） |
| **方法** | 冻结 DINOv3 ViT-L/16 + LoRA r2 + ArcFace/LogitAdjust + SupCon + MIRO + EMA，加上两个贡献点：<br>**DAS** = 载频匹配的多普勒拉伸物理增强<br>**ACR** = 载频残差分支 + 连续 log-载频 GRL 对抗头 |
| **协议** | 100 epoch，取**最后一个 epoch 的 EMA 权重**（final-EMA），**不做任何 target / source-val 挑选**，3 个种子 42/1234/31415 |
| **可训练参数** | 1.85 M（backbone 全程冻结） |

---

## 2. 环境要求

- **GPU**：一块 CUDA 显卡。论文数字是在 RTX 5090（sm_120）上跑的；30/40 系也能跑，换成对应的 CUDA 版 PyTorch 即可。
- **Python 3.10**
- **不需要联网。** DINOv3 backbone 权重（`vit_large_patch16_dinov3.lvd1689m`，1.16 GB）**已经打包在 `weights/hub/` 里**了。`baseline_v8/v8lib.py` 会把 `HF_HOME` / `HUGGINGFACE_HUB_CACHE` 指到本目录的 `weights/`，所以 timm 会直接从本地读取，全程不访问 HuggingFace。
  > 前提是**保持目录结构完整**：`weights/` 必须和 `baseline_v20/`、`tasks/` 在同一层。别单独把某个子目录拷出去用。
- 显存：评测阶段约 6–8 GB 足够。

```bash
# 建议新建一个干净环境
conda create -n xcarrier python=3.10 -y
conda activate xcarrier

# 1) 先按你自己的显卡装 PyTorch，示例是 5090 用的 cu128：
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128

# 2) 再装其余依赖
pip install -r requirements.txt
```

---

## 3. 一条命令复现

解压后 **cd 到本目录**（路径都是相对本目录解析的，位置不要错），然后：

```bash
python reproduce.py
```

几分钟跑完（首次多一次 1.2 GB 权重下载）。它会依次跑两个脚本，**不训练**，只用包里自带的 final-EMA 权重做推理。

### 应该看到的数字

```
Proposed (A_V13_GRL)，full-418 77 GHz，final-EMA，seeds 42/1234/31415

  单模型 macro-F1 .............. 0.832 ± 0.034   (每个种子 0.791 / 0.832 / 0.874)
  集成 · 多数投票 majority ....... 0.851
  集成 · logit 平均 logit-avg .... 0.857   <-- 论文的部署数字（最强系统）
  集成 · 后验平均 posterior ...... 0.856   bootstrap 95% CI [0.821, 0.889]
```

> **注意**：三种集成规则（majority / logit-avg / posterior）落点几乎一样（0.851–0.857）。这是故意做的一项检查——说明后验平均**不是**一个针对目标域调出来的技巧。

单独跑某一步：

```bash
python reproduce.py --step 1     # 只跑集成规则（出 0.832 / 0.851 / 0.857 / 0.856）
python reproduce.py --step 2     # 只跑统一评测 harness + bootstrap CI
```

输出文件：
- `EXPERIMENTSRESULT/REVISION_5090/SUPP_ABLATION/eval_out/ensemble_rules.{md,json}`
- `EXPERIMENTSRESULT/REVISION_5090/public_baseline/pb_results.{json,_auto.md}`

**参照输出**放在 `docs/expected_output/`——这是本包打包前在 RTX 5090 上实跑一遍的结果，你的输出应该和它逐位一致：

```
| Config              | Single macro-F1 | Ens majority | Ens logit-avg | Ens posterior |
| DAS+ACR (A_V13_GRL) | 0.8322 ± 0.0337 | 0.8512       | 0.8566        | 0.8562        |

#10 proposed    f1=0.8322+/-0.0337 acc=0.8357 src_f1=0.9670 gap=+0.1348 params=1.85M
                seeds=[0.791, 0.8322, 0.8735]
#11 ensemble    f1=0.8562 acc=0.8589 ci95=[0.821, 0.889]
[xcheck] proposed-family 278 vs history max|diff| = 0.0000
```

最后那行 `max|diff| = 0.0000` 是评测 harness 的自检：现场重算的结果和当初训练时记录的 `history.json` 完全一致，说明推理链路没有走偏。

---

## 4. Baseline 是什么

**完全同一套协议**跑出来的对照表：同样训 10+24 GHz、同样零样本测 77 GHz 全 418 张、同样 100 epoch、同样取 final-EMA、同样 3 个种子。差别只在方法本身。

通用 backbone（#1–#6）**只做 per-image 标准化，不加任何增强**——不翻转（时间轴编码了 Away/Towards 方向，翻转会破坏标签）、不加 DAS/HFT/SpecAugment（那些是本方法的贡献）。这样两者之间的差距才能干净地归因到方法上。

| # | 方法 | 类别 | 77 GHz macro-F1 | 77 GHz acc | 源域 val F1 | 泛化 gap | 参数量 (M) |
|---|---|---|---|---|---|---|---|
| 1 | VGG16-BN | 通用 | 0.117 ± 0.009 | 0.212 ± 0.026 | 0.983 | +0.867 | 136.38 |
| 2 | MobileNetV3-Large | 通用 | 0.146 ± 0.016 | 0.260 ± 0.015 | 0.973 | +0.828 | 4.54 |
| 3 | EfficientNet-B0 | 通用 | 0.089 ± 0.028 | 0.174 ± 0.023 | 0.913 | +0.824 | 4.34 |
| 4 | ConvNeXt-Tiny | 通用 | 0.186 ± 0.024 | 0.282 ± 0.035 | 0.981 | +0.796 | 28.22 |
| 5 | Swin-Tiny | 通用 | 0.071 ± 0.029 | 0.166 ± 0.020 | 0.980 | +0.908 | 27.92 |
| 6 | ConvNeXtV2-Tiny | 通用 | 0.150 ± 0.016 | 0.210 ± 0.015 | 0.973 | +0.823 | 28.27 |
| 7 | **RadMamba** | 雷达 SSM | 0.204 ± 0.030 | 0.245 ± 0.044 | 0.925 | +0.721 | 0.09 |
| 8 | **SelaFD-ViT-B/16** | 雷达 PEFT | 0.107 ± 0.034 | 0.191 ± 0.046 | 0.912 | +0.805 | 14.48 |
| 9 | DINOv3 ViT-L/16 + LoRA（**去掉 DAS/ACR**） | 基础模型对照 | 0.302 ± 0.031 | 0.413 ± 0.025 | 0.970 | +0.668 | 1.85 |
| 10 | **本方法（DAS + ACR）单模型** | ours | **0.832 ± 0.034** | 0.836 ± 0.032 | 0.967 | **+0.135** | 1.85 |
| 11 | **本方法 + 3 种子集成（部署行）** | ours | **0.856 / 0.857** | 0.859 | — | — | ≈3× |

> - **#10 − #9 = +0.53**，这就是 DAS+ACR 干净的跨载频贡献（同一个 backbone、同一套训练配置，只差这两个模块）。
> - **#11 是部署行，单独列，不与 #1–#9 做头对头比较**（参数量和延迟都是 3 倍）。
> - 泛化 gap = 源域 val F1 − 77 GHz F1。所有 baseline 的源域 val 都饱和在 0.91–0.98，但 77 GHz 全线崩盘 → 这不是欠拟合，是**跨载频泛化**问题。
> - **注意所有 baseline 的 worst-class F1 都是 0.00**（见 `docs/pb_results_strong_auto.md` 的逐类表）：它们在 Bend/Pick/Sit 上完全失效。本方法最差类是 Sit = 0.72。
> - RadMamba / SelaFD 是外部雷达方法，按各自原始 recipe 复现。SelaFD 官方仓库**没有 LICENSE 文件**（默认保留一切权利），这里只作学术对比，不再分发其代码。

完整表（含逐类 F1、bootstrap CI、协议设计理由）见 `docs/PUBLIC_BASELINE_RESULTS.md`、`docs/pb_results_strong_auto.md`、`docs/PUBLIC_BASELINE_DESIGN.md`。

---

## 5. 消融：两个模块各自贡献多少

完整数据见 `docs/RESULTS_SUPP.md`（21 次训练）和 `docs/GRL_RESULTS.md`。

**DAS（物理增强）是绝对主力：**

| 配置 | macro-F1 |
|---|---|
| ERM，完全不增强 | 0.146 ± 0.034 |
| 只有通用雷达增强（HFT+SpecAug，无拉伸） | 0.302 ± 0.031 |
| **只有多普勒拉伸** | **0.704 ± 0.089**（比 ERM **+55.8 pp**） |
| 无物理依据的随机 jitter | 0.518 ± 0.030 |
| 完整 DAS | 0.767 ± 0.076 |

→ 拉伸必须**按载频匹配**才有效：物理 jitter 0.518 ≪ 载频匹配拉伸 0.704。

**ACR（对抗载频残差）在 DAS 之上再加 +6.5 pp：**

| 配置 | macro-F1 | Δ |
|---|---|---|
| 只有 DAS | 0.767 ± 0.076 | — |
| + 只加残差分支 | 0.814 ± 0.021 | +4.7 |
| + 只加 GRL | 0.826 ± 0.016 | +5.9 |
| **+ 完整 ACR** | **0.832 ± 0.034** | **+6.5** |
| + 换成普通离散域对抗（3-bin CE） | 0.796 ± 0.034 | +2.9 |

→ **连续 log-载频回归比普通离散域对抗好 +3.6 pp**（配对 bootstrap 95% CI [+1.91, +5.31]，P(Δ>0)=1.000）。这是方法新颖性的主要支撑点。

**统计显著性**（配对分层 bootstrap，B=10000，N=418）：

| Δ | 观测值 | 95% CI | P(Δ>0) |
|---|---|---|---|
| DAS+ACR 单模型 − DAS 单模型 | +6.55 pp | [+4.69, +8.37] | 1.000 |
| DAS 单模型 − jitter 单模型 | +24.90 pp | [+21.47, +28.32] | 1.000 |
| 集成 − 单模型 | +2.40 pp | [+0.89, +3.96] | 0.999 |
| 连续对抗 − 离散对抗 | +3.57 pp | [+1.91, +5.31] | 1.000 |

---

## 6. 目录结构

```
.
├── README.md                    <- 本文件
├── reproduce.py                 <- 一条命令复现入口
├── requirements.txt
├── paper/
│   └── Zero-Shot Cross-Carrier Transfer for Radar.pdf
├── tasks/known_people_unknown_freq/
│   ├── dataset/{train,val,test}/    <- 1930 张时频图（jet 色彩，224×224）
│   ├── manifest/                    <- train/val/test 划分 CSV（确定性划分）
│   └── summary.json
├── baseline_v20/                <- 方法主体：config.py / v9_2_1lib.py / train.py
│   ├── ensemble_rules.py            <- 【复现 0.857 的脚本】
│   ├── paired_bootstrap_ci.py       <- 配对 bootstrap 置信区间
│   ├── aggregate_ablation_finalema.py
│   └── ...
├── baseline_v9/v9lib.py         <- 被 v9_2_1lib 依赖
├── baseline_v8/v8lib.py         <- 被 v9lib 依赖；这里设置 HF_HOME -> ./weights
├── pool_protocol.py
├── weights/hub/                 <- 预置的 DINOv3 ViT-L/16 backbone（1.16 GB，离线用）
├── EXPERIMENTSRESULT/
│   ├── v15_v18_common_train.py  <- 训练主循环（要重训才需要）
│   └── REVISION_5090/
│       ├── A_V13_GRL/seed{42,1234,31415}/
│       │   ├── checkpoints/pool_ep100_ema.pt   <- 最强系统权重（各 7.1 MB）
│       │   ├── reports/                        <- 训练历史 history.json、逐 epoch 记录
│       │   └── manifests/
│       └── public_baseline/     <- baseline 评测 harness（pb_*.py）+ runs_strong 结果
└── docs/
    ├── GRL_RESULTS.md               <- ACR/GRL 完整结果与机制分析
    ├── RESULTS_SUPP.md              <- 21 次消融训练的完整结果
    ├── ABLATION_REPORT.md
    ├── PUBLIC_BASELINE_RESULTS.md   <- baseline 表 + 协议
    ├── PUBLIC_BASELINE_DESIGN.md
    ├── pb_results_strong_auto.md    <- 自动生成的 baseline 表（含逐类 F1）
    ├── pb_results_strong.json
    └── DATA_SOURCES.md              <- 数据来源与授权
```

---

## 7. 想从头重训（可选，约 6 小时 / 3 个种子）

```bash
# 单个种子，约 2 小时（RTX 5090）
V13_GRL_WEIGHT=0.3 V13_GRL_TARGET=shown V921_USE_DAS=1 \
  V921_RUNS_DIR_TAG=A_V13_GRL_repro python EXPERIMENTSRESULT/v15_v18_common_train.py --seed 42
```

方法的所有开关都是环境变量驱动的（定义在 `baseline_v20/config.py`），不用改代码：

| 环境变量 | 作用 | 论文配置 |
|---|---|---|
| `V921_USE_DAS` | 开启 DAS 物理增强 | `1` |
| `V921_DAS_MODE` | DAS 调度：`curriculum` / `fixed_full` / `fixed_narrow` / `jitter` | `curriculum` |
| `V13_GRL_WEIGHT` | ACR 对抗头权重，`0` = 关闭 | `0.3` |
| `V13_GRL_TARGET` | 对抗回归目标：`shown`（= log f_src + 残差）/ `base` | `shown` |
| `V921_FAST_GPU` | GPU 增强快路径（约 2.2× 加速） | `1` |

⚠️ **重训的数字会有约 ±0.05/种子的抖动**（GPU 非确定性 + 数据量小，644 张训练图）。要**逐位复现论文表格**请用自带权重跑 `reproduce.py`，不要重训。

---

## 8. 已知限制（请照实转述，不要夸大）

1. **只跨载频，不跨人。** 训练集和测试集是同一批受试者。任何地方都不要写成 "unseen-subject" 或 "cross-subject"。
2. **载频探针机制是被证伪的（negative result）。** GRL 并没有让特征对载频不可分——`z_cls → 载频` 的线性探针一直是 0.999，加大对抗力度会先把类别信息压垮（0.612）再动探针。起作用的机制是**连续多普勒尺度不变性**，不是"擦掉载频信息"。**论文不能声称载频不变性 / probe→chance**。详见 `docs/GRL_RESULTS.md` §3 和 §7。
3. **只带了本方法的权重。** A_REF、E1_noDAS、E1_jitter、E2_narrow、E2_full 这 5 组消融权重原盘已删，无法重新导出——所以 `reproduce.py` step 1 会对它们打印 `[skip]`，step 2 会对 #1–#9 打印 `NOT RUN`。这些行的数字在 `docs/` 里有完整记录（是从当时的训练日志和 JSON 里来的），但**在本包内无法重新计算**。本方法自己的 0.832 / 0.857 / CI 全部可以现场重算。
4. **单种子方差偏大**（±0.034，个别消融行到 ±0.089）。这是 644 张训练图 + GPU 非确定性的结果，集成把方差从 0.076 压到 0.034 正是动机之一。
5. **评测 harness 已交叉验证**：本方法在 278 张子集上的重算值 vs 训练时记录的 `history.json`，`max|diff| = 0.0000`。

---

## 9. 数据来源与授权

数据集 = 美国 **University of Alabama CI4R** 跨频段微多普勒数据集（XeThru 10 GHz / Ancortek 24 GHz / TI 77 GHz，共址采集），在 24 GHz 通道上并入了一批 **加拿大国家研究委员会（NRC Canada）** 的 24 GHz 采集数据，两者在训练中联合使用。

- CI4R 原始仓库：https://github.com/ci4r/CI4R-Activity-Recognition-datasets
- 详细来源与注意事项：`docs/DATA_SOURCES.md`
- 代码按 `LICENSE`（MIT）发布；**数据集是第三方资产，仍受 CI4R 自身条款约束**，使用时请引用 CI4R 数据集。
- `weights/hub/` 里的 DINOv3 backbone 是 **Meta 的预训练权重**（经 timm 分发），受其自身许可约束，这里只是为了离线可用而随包附带。
- 这份包是**内部分享**用的，请不要对外再分发数据和权重部分。

引用信息见 `CITATION.cff`。
