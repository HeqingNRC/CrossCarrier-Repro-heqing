# 复现与三方向扩展

本次在原包副本上修改，原始 `CrossCarrier_Repro.zip` 保持不变。新的操作入口是本指南与 `hpc/README_zh.md`；原 README 保留供参考，其中旧重训命令和默认配置不等于本次固定的实验配方。

## 1. 已经做了什么

已在 RTX 4070 Laptop 8GB 上用自带三个 final-EMA checkpoint 重新推理，完整评估原 418 张 77 GHz 图片。Python 3.12.13，PyTorch 2.11.0+cu128，timm 1.0.26，PEFT 0.18.1，BF16，评估 batch 8。包内没有记录原作者的 PEFT/Transformers 版本，本次环境属于实际验证过的兼容环境，不是原环境逐项克隆。

| 指标 | 包内参考 | 本次重新计算 |
|---|---:|---:|
| 单模型 macro-F1，三 seed 均值±总体标准差 | 0.8322±0.0337 | 0.8330±0.0346 |
| 多数投票 macro-F1 | 0.8512 | 0.8512 |
| Logit 平均 macro-F1 | 0.8566 | 0.8566 |
| Logit 平均 accuracy | 0.8589 | 0.8589 |
| 概率平均 macro-F1 | 0.8562 | 0.8562 |
| 概率平均 bootstrap 95% CI | [0.821, 0.889] | [0.8213, 0.8887] |

两套评测入口结果一致；278 张核对集与原训练 history 的 macro-F1 差值为 0。完整 418 张的第三个 seed 有微小差异：原约 0.8735，本次约 0.8758。不同硬件、批量和数值实现可能改变临界样本预测，尚未单独定位原因，不能称为所有单模型数字逐位一致。

**这是已训练模型的推理复现。三个方向的正式 100 epoch 重训尚未完成，也尚未登录/提交 Sockeye。** 本地另外执行短程 GPU 训练、EMA 保存/重新加载、完整目标集评估来验证运行链路；这些 smoke 输出不作为识别性能结论。测试状态与日志见交付目录中的 `validation.json` 和 `reproduction/`。

## 2. 为什么不能只改频率列表

| 文件/入口 | 修改与用途 |
|---|---|
| `prepare_cross_frequency.py` | 合并原数据索引、保留七类、识别录制编号与图片副本，生成三方向 manifest 和审计报告 |
| `baseline_v20/config.py` | 允许动态 source/target 频率、task/data 路径和精度；保留原入口默认行为 |
| `cross_frequency.py` | 新统一入口，明确 target、protocol、variant、seed、100 epoch 和输出目录 |
| `EXPERIMENTSRESULT/v15_v18_common_train.py` | 复用原训练损失；新增只读 source 的训练分支，最后一轮无条件保存 EMA |
| `baseline_v20/v9_2_1lib.py`、`baseline_v8/v8lib.py` | 通用频段与精度支持；新入口的 HFT 使用可播种 RNG，修复原实现每张图新建未播种 RNG 的问题 |
| `reproduce.py`、`portable_eval.py` 与两个 evaluator | 小批量评测、CPU/GPU 精度适配、独立输出、离线缓存、检查 adapter 权重是否完整 |
| `requirements-portable.txt` | 补齐 PEFT/Transformers/tqdm 等依赖；PyTorch 与 CUDA 单独按 GPU 安装 |
| `aggregate_cross_frequency.py` | 汇总三 seed 与两种集成，并计算 proposed-control 的配对提升；拒绝混合划分/配方、重复 seed、smoke |
| `hpc/` | Sockeye 参数化 Slurm 脚本、资源查询、GPU 检查和中文指南 |

新入口保留原 DINOv3 ViT-L/16 + LoRA r2、CE/SupCon/MIRO/ACR 损失及训练时 kin+sensor logits、评估时 kin-only logits。载频域编号从真实 source 频率计算，例如 `24+77→10` 的两个域对应 24 和 77 GHz，不能把 77 GHz 图片重命名为 10 GHz。

DAS 采用训练历史中实际记录的三阶段：epoch 1–8 为 [10,24] GHz/p=0.35；9–24 为 [10,50]/p=0.70；25–100 为 [12,95]/p=1。先在所有方向保持这套配方，形成可比较的起点。它原本针对向高频外推设计，不保证向 10 GHz 迁移同样有效。

新训练没有 target early stopping、target 选 checkpoint 或按 target 分数挑 seed。模型固定用最后 epoch 的 EMA，训练结束才打开 target manifest/image。原包遗留的 `best_pool`、`best_sourcequalified77` 等目标成绩诊断逻辑不参与新入口。

## 3. 两种划分必须分别报告

原包共 1,930 张、11 类。沿用论文任务，只研究共同七类：Away、Bend、Kneel、Pick、SStep、Sit、Towards。

原七类 source train/val 有 **121 个共享录制编号、8 组完全相同图片**。七类全数据有 26 个多余副本：10 GHz 19 张、77 GHz 7 张。这样会影响 source validation 的独立性，也影响重复图片的样本权重。

因此保留原划分核对旧结果，另建按 subject+class+timestamp 分组且按文件哈希去重的统一实验划分；train/val 中相同录制与相同图片的交集均为零。

| Protocol | Source→target GHz | Train | Source val | Target test |
|---|---|---:|---:|---:|
| `legacy` | 10+24→77 | 644 | 162 | 418 |
| `grouped` | 10+24→77 | 634 | 153 | 411 |
| `grouped` | 10+77→24 | 620 | 152 | 426 |
| `grouped` | 24+77→10 | 672 | 165 | 361 |

所有训练 seed 使用同一组 manifest。生成划分的 seed 42 与训练模型的 seed 是两件事，不要每次训练重新分数据。

**仍然是 known-person、同步动作录制的跨频段任务。** Source 和 target 之间仍共享同步录制，grouped 只隔离 source train/val，不能由此声称跨未见人员、录制或 session 的泛化。新的 grouped→77 成绩也不能直接当作原 418 张 benchmark 的重现成绩。

## 4. 本地与集群共用的命令

在 `CrossCarrier_Repro/` 根目录、已激活兼容 Python 环境后：

```bash
# 标准库即可运行；只准备一次，生成全部新划分和原始77对照
python prepare_cross_frequency.py --protocol both

# 原模型推理复现，默认batch8，结果与包内参考分开保存
python reproduce.py --output-root output/reproduction --batch-size 8 --precision auto

# 不加载模型或目标图片，先确认三个方向的source索引
python cross_frequency.py train --target 77 --run-dir output/check77 --dry-run
python cross_frequency.py train --target 24 --run-dir output/check24 --dry-run
python cross_frequency.py train --target 10 --run-dir output/check10 --dry-run

# 三个方向的正式命令；每个再以1234和31415重复
python cross_frequency.py train --target 77 --protocol grouped --seed 42 --epochs 100 --run-dir output/grouped/proposed/target77/seed42
python cross_frequency.py train --target 24 --protocol grouped --seed 42 --epochs 100 --run-dir output/grouped/proposed/target24/seed42
python cross_frequency.py train --target 10 --protocol grouped --seed 42 --epochs 100 --run-dir output/grouped/proposed/target10/seed42
```

这些训练命令在 Sockeye 上应由 Slurm GPU 作业执行，不能直接在登录节点运行。Linux 可以加 `--num-workers 4`；本机 Windows 使用默认 0。训练默认 batch 16；若改为 8，需要在三个方向及对照组保持一致并记录。V100 用 FP16+GradScaler，`--precision auto` 也会按原生硬件能力选择。

若需要在原 source 划分重训旧方向，使用：

```bash
python cross_frequency.py train --target 77 --protocol legacy --seed 42 --epochs 100 --run-dir output/legacy/proposed/seed42
```

这是在原数据划分上的重新训练尝试。新入口修复了 HFT 随机数播种，且原包缺少完整环境与 fast-GPU 运行配置，因此不能保证重新训练逐位还原学弟 checkpoint。

正式训练结果位于 `RUN_DIR/checkpoints/final_ema.pt`、`RUN_DIR/reports/final_target_metrics.json` 和 `final_target_predictions.npz`。`RUN_DIR/run_config.json` 记录有效配方和 manifest 指纹。

如只先训练而暂缓测试，加 `--skip-final-eval`；稍后用匹配的 target/protocol/variant 运行：

```bash
python cross_frequency.py eval --target 10 --protocol grouped --run-dir output/grouped/proposed/target10/seed42
```

当前新入口**不支持断点续训**，会拒绝覆盖已有训练目录。先用相同正式 batch 做短运行估算速度，再根据 allocation 的 walltime 限制安排 100 epoch；不要把 smoke 或短运行接着当正式结果。原 resume bundle 不包含完整的 adversary/scaler/sampler 状态，所以没有沿用不完整恢复。

## 5. 怎样判断是否 improve

至少比较两个 variant，每个方向使用 seed 42、1234、31415：

- `proposed`：DAS + ACR，默认。
- `no_das_acr`：同 DINOv3/LoRA、通用 HFT/SpecAugment、训练预算和划分，关闭 DAS 与载频残差/对抗损失。

使用 `--variant no_das_acr` 并更换 run-dir 重训对照。三个方向×三 seed×两种方法，共 **18 个正式训练任务**；先跑 proposed 九个任务检查资源与稳定性，再提交对照九个任务。不要拿新方向的现有旧77 checkpoint 当作训练后成绩。

全部完成后汇总一个包含这些 run 的目录：

```bash
python aggregate_cross_frequency.py --runs-root output/grouped --output output/summary
```

汇总给出单模型 accuracy/macro-F1 的 mean±std、每个 seed、logit/probability ensemble 和 proposed-control 提升百分点。单模型与三模型集成应分开比较。训练失败的 seed 必须补齐，不能挑表现好的 seed；脚本会拒绝同一 seed 多个候选和不完整 seed 组。

后续若低频方向较差，可以研究双向 DAS 范围/调度，但应先固定消融设计，并依据 source-side validation 或内部伪目标域选择超参数，不能根据真实目标测试集反复挑参数。

## 6. Sockeye 上操作

完整步骤见 `hpc/README_zh.md`。交付的 `CrossCarrier_Sockeye_ready.tar.gz` 包含代码、图片、已有 checkpoint 与离线 backbone，已使用 Linux 可正常解包的目录形式；不包含本机 Windows venv。

1. 用 Globus 或学校允许的 SCP/SFTP 上传 tar.gz 到自己的 project 存储。
2. 在 Linux 解压：`tar -xzf CrossCarrier_Sockeye_ready.tar.gz`，然后 `cd CrossCarrier_Repro`。
3. 按 HPC 指南建立 Linux 环境并设置 `ENV_SETUP`。
4. `cp hpc/config.example.sh hpc/config.sh`，填写 account、GPU partition、PROJECT_DIR、OUTPUT_ROOT、ENV_SETUP。
5. `bash hpc/submit.sh preflight`：先确认 GPU/环境。
6. `bash hpc/submit.sh repro`：核对原 checkpoint。
7. `bash hpc/submit.sh train`：提交三个方向×三个 seed 的数组，默认同时只跑一个任务。

Slurm 输出在 `OUTPUT_ROOT/slurm/`；模型结果在 `OUTPUT_ROOT/train/job-ARRAY_ID/VARIANT/targetN/seedS/model/`。汇总可对 `OUTPUT_ROOT/train` 执行，但若有重复 seed 的正式重跑，应按预先规定选择完整实验目录，不能依 target 成绩挑选。

原本地 torch 2.11+cu128 构建不支持 V100；HPC 指南提供经官方架构文档核查、尚未上集群实测的替代环境。具体 SSH 主机、allocation、partition 和路径必须取自你的账户，不能凭空假设。
