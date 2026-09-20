# CrossCarrier: radar recognition across 10, 24 and 77 GHz

基于原作者 Zirui Lin 提供的 CrossCarrier 复现代码，增加三个载频方向的统一训练入口：

- 10 + 24 → 77 GHz
- 10 + 77 → 24 GHz
- 24 + 77 → 10 GHz

七类活动识别；已知人员与同步动作录制的跨频率任务，不是跨未见人员/录制的泛化实验。

## 仓库包含什么

包含模型/训练/评测源码、分组划分生成器、三种子结果汇总、HPC 模板和测试。
**不包含 radar 图片、manifest 数据索引、DINOv3 权重、已有 checkpoint、私人运行结果或本地 Python 环境。**
因此仅 `git clone` 后不能直接训练；先按下方步骤把已获授权的复现包资产接入本地目录。

原作者版权信息保留在 `LICENSE`，引用信息见 `CITATION.cff`。后者仍有原作者留下的 TODO，请在正式发表前核对。
原包的第三方数据/权重说明不能替代它们各自的使用条款；代码仓库不分发这些资产。数据来源说明见 `docs/DATA_SOURCES.md`。

## 首次运行：接入私有资产

从完整复现包复制以下内容到本代码目录对应位置，目录层级保持一致：

1. `tasks/known_people_unknown_freq/`：图片与原始 train/val/test manifest。
2. `weights/`：离线 DINOv3 backbone。
3. `EXPERIMENTSRESULT/REVISION_5090/A_V13_GRL/`：只有运行旧模型推理复现时需要。

它们已被 `.gitignore` 排除。不要把整个 1.25 GB 压缩包放进 Git 提交。

在 Sockeye 上也可用符号链接接入已解压的资产。在**这个代码目录**内执行下面命令；`ASSETS` 指向实际存在的原包根目录：

```bash
ASSETS=/scratch/st-zliu-1/heqingz/CrossCarrier_Repro
ln -s "$ASSETS/tasks" tasks
ln -s "$ASSETS/weights" weights
ln -s "$ASSETS/EXPERIMENTSRESULT/REVISION_5090/A_V13_GRL" EXPERIMENTSRESULT/REVISION_5090/A_V13_GRL
```

上面的链接仅在目标位置尚不存在时创建；已有真实目录可直接保留使用。不要用 `ln -sf` 覆盖已有资产。
多人/多实验共享资产时，只运行一次划分准备，训练过程中不要修改共享 manifest。

## 环境与数据划分

先按 GPU 安装匹配的 PyTorch/torchvision，再安装其他依赖。V100 的候选环境和 Slurm 操作见 `hpc/README_zh.md`，尚未在 Sockeye 实测。

```bash
python -m pip install -r requirements-portable.txt
python prepare_cross_frequency.py --protocol both
python -m unittest discover -s tests
```

纯代码 checkout 也能运行测试：未提供私有数据时，3个数据集成测试会明确跳过；接入资产并生成两种划分后，才执行完整30项测试。

原七类 source train/val 有121个共享录制编号与8组相同图片。`grouped` 去重并隔离 source train/val 的录制组；`legacy` 仅支持原77GHz目标划分。
源频段与目标频段仍保留同步录制的对应关系。

| Protocol | Target GHz | Train | Source val | Target test |
|---|---:|---:|---:|---:|
| legacy | 77 | 644 | 162 | 418 |
| grouped | 77 | 634 | 153 | 411 |
| grouped | 24 | 620 | 152 | 426 |
| grouped | 10 | 672 | 165 | 361 |

## 训练与评估

```bash
# 使用已有权重复现原10+24→77结果
python reproduce.py --batch-size 8 --precision auto --output-root output/reproduction

# 新方向；把target替换为77、24或10；seed使用42、1234、31415
python cross_frequency.py train --target 10 --protocol grouped --variant proposed --seed 42 --epochs 100 --run-dir output/grouped/proposed/target10/seed42

# 同backbone对照：关闭DAS/ACR
python cross_frequency.py train --target 10 --protocol grouped --variant no_das_acr --seed 42 --epochs 100 --run-dir output/grouped/no_das_acr/target10/seed42

python aggregate_cross_frequency.py --runs-root output/grouped --output output/summary
```

新入口在训练期间不读取目标图片/标签，固定取最后epoch的EMA，之后才评估目标集。默认保留原DAS三阶段范围与100epoch预算，HFT改用已播种随机数。训练时使用原kin+sensor logits，评估使用kin-only logits。
训练批量/精度/配方变更应对所有比较方向与方法保持一致。当前不支持断点续训，拒绝覆盖非空run-dir。

## Sockeye 与 GitHub 配合

GitHub管理代码版本；Sockeye已有的图片/权重保留在scratch，`git pull`不会下载这些资产。
从GitHub克隆代码后，按上面的方式接入数据，将 `hpc/config.sh` 中 `PROJECT_DIR` 改为**克隆后的代码目录**，`OUTPUT_ROOT` 设置为独立结果目录。
个人 `hpc/config.sh` 与激活脚本保持本地，不提交GitHub。不要在运行任务期间更新其代码或配置。

## 当前验证状态

2026-09-20，本机RTX4070Laptop8GB：30项测试通过；三个方向短程GPU训练/EMA保存与评估检查通过，并验证FP16+GradScaler。
原full418已有checkpoint：logit平均集成macro-F1=0.8566、accuracy=0.8589，与包内参考一致；单模型均值0.8330，参考0.8322，存在微小数值差异。
正式100epoch三方向训练尚未完成；未在Sockeye提交作业。
