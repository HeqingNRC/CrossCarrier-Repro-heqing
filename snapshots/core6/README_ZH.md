# CoreMovements6：完整方法与消融的独立重训

## 本包要做什么

将本轮 Original7 已运行的 7 种补充消融和 3 个 anchor，共 10 个配置，
在 CoreMovements6 的三个留出频率方向上各跑 3 个种子：90 次全新正式训练。
这是新的六分类实验。旧 Core6 分数、七分类 checkpoint、当前 Original7 作业，均不会填入本次结果。
本包没有替用户提交任何 Sockeye 作业，也没有在制作环境里完成实际 DINOv3 GPU 训练。

本包基于找回的 CrossCarrier_Controlled_v1，以及其内保存并校验过的学弟原始源码。
`evidence/` 中模型、增强和损失原文件保持原样；`train_controlled.py` 是本次新的控制器。
它使用原训练循环的 forward/loss AST，不运行原来的 target-best / pool 选择控制逻辑。
新控制器支持 V100 的 FP16 + GradScaler，固定第100轮 EMA，无 source-best 或 target-best 挑选。
这不等于逐位重放当年机器和随机数执行轨迹，因此完整 proposed 也在本批重新训练。

## 数据和协议

六类顺序：Away / Bend / Kneel / Pick / Sit / Towards。

| 源频率 -> 目标 | Source train | Source val | Target test |
|---|---:|---:|---:|
| 24+77 -> 10 GHz | 677 | 102 | 73 |
| 10+77 -> 24 GHz | 496 | 74 | 126 |
| 10+24 -> 77 GHz | 677 | 102 | 73 |

原始清单已经包含在 `manifests/core6_source/`。
这些清单来自之前 Core6 的 split_manifest，不使用七分类 grouped manifest，不重新随机划分。
原清单的图片根目录为：
`/arc/project/st-zliu-1/heqingz/datasets/domainGenDataset_core_movements`

prepare 只检查这三个清单中列出的 1336 个 PNG 路径，读取文件哈希和图像哈希，不全盘搜索。
如果原目录被移动，prepare 会保存 `preparation_failure.json` 并停止；可通过
`--core-root /实际目录` 显式指定包含 `10GHz/24GHz/77GHz` 的图片根目录。
不从不明 H5 重新划分，不用七类模型删一列来代替六类重训。

`--preserve-legacy-splits` 明确保留历史图像级划分及已记录的同标签重复风险；不改图、不删样本。
跨类别完全相同的输入会导致准备阶段停止。重复风险会写入协议和汇总。
本实验不能据此声称受试者独立、录制独立或物理坐标标定已验证。

## 10 个配置

| recipe | 与本轮七分类配置的对应关系 |
|---|---|
| no_image_aug | DAS_ERM：关闭 DAS/HFT/SpecAugment；其余分类、SupCon、特征锚定保留 |
| radar_aug | E1_noDAS：保留 HFT/SpecAugment，关闭 DAS/ACR；matched control |
| stretch_only | DAS_stretch_only：只开多普勒拉伸，不开通用增强或 ACR |
| das_only | A_REF：完整 DAS，没有 ACR |
| residual_only | ACR_Lfreq_only |
| grl_only | ACR_GRL_only |
| discrete_acr | ACR_discrete：3-bin CE，仍保留残差 |
| proposed | A_V13_GRL：完整 DAS + 残差 + continuous GRL |
| wide_acr | SCHED_fixedfull_ACR，15–140 GHz |
| narrow_acr | SCHED_fixednarrow_ACR，10–30 GHz |

每项100 epochs，batch16，eval batch8，FP16，seeds42/1234/31415。
所有配置共用同一个六分类模型库、图像处理、优化器配方、预训练权重和固定 split。
旧 decorrelation 和 V15/V15R 均关闭，不启用 Fishr/SWAD/FDCVNN 或其他 backbone 基线。
DAS 范围照原配方迁移到三方向，不根据目标结果调整。
fixed-range 与 curriculum 的比较同时改变了范围和调度，不应仅称为纯调度消融。

## 上传和运行

本地 PowerShell（不是 SSH 内）：

```powershell
scp "$env:USERPROFILE\Downloads\CrossCarrier_Core6_FullSuite_20260924.tar.gz" heqingz@sockeye.arc.ubc.ca:/scratch/st-zliu-1/heqingz/
```

Sockeye：

```bash
cd /scratch/st-zliu-1/heqingz
tar -xzf CrossCarrier_Core6_FullSuite_20260924.tar.gz --keep-old-files
cd CrossCarrier_Core6_FullSuite_20260924
PY=/scratch/st-zliu-1/heqingz/envs/xcarrier_20260922/bin/python
"$PY" suite.py prepare --preserve-legacy-splits
```

prepare 使用现有环境，不安装、不升级包。只链接 fresh 资产包的预训练 DINOv3 weights，
不链接任何任务 checkpoint 或实验结果目录。代码包必须使用 Python 3.10 / torch2.5.1 / torchvision0.20.1。
CPU 数学测试成功且数据准备通过后出现 `PREPARE_OK`。

然后提交：

```bash
"$PY" suite.py submit --max-concurrent 3
```

它创建本次独有的 `batches/batch-UTC时间-随机后缀/` 并保存：
- 新 smoke 数组 Job ID（12个短程测试，最大并发1）；
- 新 formal 数组 Job ID（90个正式 train+eval，最大并发3）；
- 不可与旧结果混用的 plan.json、experiment_ledger.csv、submission.json。

12个 GPU 测试覆盖10种配置以及3个目标方向。每个真实 DINOv3/LoRA 做两次成功 optimizer update、
检查活动分支梯度、序列化恢复和 EMA API，不读取目标图、不生成科研准确率。
正式数组依赖整个新 smoke 数组 `afterok`。任一测试失败，90次训练不会放行。
formal 数组索引：0–29 target10、30–59 target24、60–89 target77；组内按上表 recipe 顺序，每项3个种子。
不要修改代码、runtime.json 或 prepared 文件；入口会校验哈希。
重复执行 submit 默认拒绝再次提交；确实需要完整另跑一批时，需显式加 `--new-batch`。
现有13053829/13053830等其他项目作业不被修改，也不是本批的依赖。

## 评估和报告

每个正式任务完成100 epochs后：无条件保存 final_ema.pt，写入选择冻结记录，
然后才读取本方向的 target test PNG，在同一张已分配 GPU 上评估并保存逐样本 logits。
不在登录节点启动 DINOv3 推理，不另读旧 checkpoint。

最后一个完成的任务会尝试进行一次全批 CPU 汇总。也可以随时手动核验：

```bash
cd /scratch/st-zliu-1/heqingz/CrossCarrier_Core6_FullSuite_20260924
/scratch/st-zliu-1/heqingz/envs/xcarrier_20260922/bin/python suite.py report
```

report 只读取 CURRENT_BATCH.txt 指向的本次批次，不扫描全 scratch。
先显示本批实际 Job ID / sacct，再校验 run身份、100轮history、有效配方、split、checkpoint哈希、预测顺序和指标。
只有每组3个规定种子齐全且有效才显示该组均值；缺失项单独列明，不补旧分数。
全部90个完成后，`summary/summary.json` 的状态为 `COMPLETE_90_RUNS`，并写 `COMPLETE90.json`。

输出：

```
batches/<本批次>/
  submission.json
  plan.json
  experiment_ledger.csv
  logs/
  smoke/
  runs/<index_target_recipe_seed>/
    run_manifest.json
    effective_config.json
    history.json
    gradient_check.json
    checkpoints/final_ema.pt
    selection_frozen_before_target.json
    predictions/final_ema.npz
    predictions/final_ema.csv
    predictions/final_ema.json
    DONE.json
  summary/
    summary.md
    summary.json
    summary.csv
    per_seed.csv
    paired_deltas.csv
    confusion_matrices.csv
```

mean ± SD 使用三个规定种子的 population SD；logit、posterior、majority 集成分开记录。
paired_deltas 给同seed treatment-control差值，不自动宣称统计显著。
summary.json 同时保存逐类 precision/recall/F1、混淆矩阵和集成结果。
没有恢复原始 optimizer/RNG 状态的精确续训功能。失败/超时必须明确记录，不冒充完成。
如果90个任务未齐，report会标明 INCOMPLETE_OR_INVALID。

## 构建时验证边界

见 `BUILD_VALIDATION.json`：语法检查、7项协议合同检查、原始14配方替代编码器CPU反向传播、
合成90任务汇总与旧Job拒绝测试通过。它们不等于真实DINOv3 GPU测试。
实际PNG是否仍存在、DINOv3缓存能否加载、V100显存与FP16梯度是否正常，由Sockeye prepare / GPU smoke验收。
本文中没有提供任何这次Core6重训的准确率，因为尚未运行。

源码许可证保留在 `evidence/ORIGINAL_LICENSE`，SWAD相关原工具文件的许可证保留在 vendor/。
`archive/` 是找回版本的参考材料，不是本轮要执行的旧提交脚本。
