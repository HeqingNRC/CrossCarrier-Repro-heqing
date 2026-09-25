# CrossCarrier_Controlled_v1

**版本状态：源码审计 + CPU 数学测试通过；真实 DINOv3 GPU 训练尚待 smoke。**

本工具使用你现有的 `xcarrier` 环境，不安装包，不改原仓库，不覆盖旧 checkpoint 或数据。
所有新输出保存在这个新目录。请从本目录提交作业。

## 第一步：在 Sockeye 检查数据和冻结清单

```bash
cd /scratch/st-zliu-1/heqingz/Reproduce_Zirui/CrossCarrier_Controlled_v1
/scratch/st-zliu-1/heqingz/envs/xcarrier/bin/python prepare.py
cat prepared/summary.json
```

第一次会生成 `prepared/`；拒绝覆盖同名目录。重新审计需要 `--out prepared_2`，后续训练也要明确传 `--prepared prepared_2`。

`prepare.py` 从历史原始 manifest 恢复七分类和六分类各折。它读取图片并计算文件/RGB/224输入哈希，记录跨split重复，不删除或重分数据。不创建模型、不占用GPU。

若 Core6 的原始图片根目录已经搬迁，可显式传 `--core-root /新根目录`；它只替换根目录，保留 frequency/class/basename 后缀，不做模糊匹配。

## 第二步：只提交 GPU smoke

```bash
sbatch jobs/00_smoke.sbatch
```

此 job 顺序执行当前环境下的 CPU 自测、七分类8步训练检查、六分类8步训练检查。它不评测 target，不产生正式结果。输出是：

```
xc_audit_smoke_JOBID.out
xc_audit_smoke_JOBID.err
runs/smoke_original7_JOBID/SMOKE_PASS.json
runs/smoke_core6_JOBID/SMOKE_PASS.json
```

日志最后出现 `[ALL PASS]`，且两份 `SMOKE_PASS.json` 存在，才进入正式单种子试跑。

默认精度 bf16，沿用原配方；不做静默降级。日志区分GPU原生BF16能力与软件支持，不把BF16张量可创建等同于原生Tensor Core支持。若GPU运行时明确报精度不支持，应先记录错误。经确认采用fp32测试时：

```bash
XC_PRECISION=fp32 sbatch --export=ALL jobs/00_smoke.sbatch
```

这会被记录成精度变体，不直接当作原bf16结果。

## 第三步：受控单种子训练（先看数据审计，再操作）

生成计划不会提交：

```bash
python make_jobs.py --dataset original7 --target 77GHz --recipes proposed \
  --seeds 42 --out plans/original7_seed42
```

确认 `plans/original7_seed42/ledger.csv` 后才执行：

```bash
sbatch plans/original7_seed42/submit.sbatch
```

正式计划默认12小时上限是资源申请上限，不是耗时预测。训练没有自动resume；先测一个epoch/单种子的实际用时再调整。日志含有效超参数、source/target协议与版本快照。

如果图片哈希确认了当前活跃split之间的重复，而任务明确是按旧数据协议做历史复现，必须在计划生成时加 `--acknowledge-duplicate-risk`。这一开关仅记录对legacy协议的显式保留，不会使数据变成无重复、受试者独立或无泄漏。新干净协议需另起一组实验。

## 第四步：原七分类三个种子、消融；再迁移 Core6

单种子桥接通过后，生成主方法三种子计划：

```bash
python make_jobs.py --dataset original7 --recipes proposed --out plans/original7_main
```

完整 Table II + schedule 控制组：

```bash
python make_jobs.py --dataset original7 --recipes all --out plans/original7_ablations
```

这会生成42次训练，默认最多同时运行1个；**不会自动提交**。已经完成的主方法seed不要在新计划里重复跑，必要时用 `--recipes` 明确只生成未跑配置。

六分类先做10+24→77：

```bash
python make_jobs.py --dataset core6 --target 77GHz --recipes proposed --out plans/core6_main77
```

六分类其余两折：

```bash
python make_jobs.py --dataset core6 --target 10GHz --recipes proposed --out plans/core6_main10
python make_jobs.py --dataset core6 --target 24GHz --recipes proposed --out plans/core6_main24
```

100 epochs/原三种子用于复现配方；如果要与旧50epochs/10seeds结果做严格训练预算控制，另起清楚标记的50epochs/相同seed矩阵，不混报。

## 历史训练控制器回放（明确允许 target 诊断）

`replay_original.py` 直接调用原控制器，恢复主配方、epochs100，保留其target诊断；将最终epoch EMA另存，避免source阈值导致没有final文件。只保存最终epoch dump，不保存全部100份。

它必须在GPU Slurm作业内运行：

```bash
python replay_original.py --recipe proposed --seed 42 \
  --out runs/legacy_proposed_seed42_run1 --allow-target-diagnostics
```

若需保留已核实的旧重复数据协议，同样加 `--acknowledge-duplicate-risk`。这条回放与 `train_controlled.py` 的target-blind新结果分别报告。不要直接在登录节点执行；代码会拒绝无GPU训练。

## Fishr + SWAD

先运行扩展 smoke：

```bash
sbatch jobs/01_dg_smoke.sbatch
```

lambda1、anneal0在这里仅用于尽早测试梯度流，**没有据此选定正式超参数**。

Core6主配方基线正式跑通后：

```bash
python make_jobs.py --dataset core6 --target 77GHz --recipes proposed --swad \
  --out plans/core6_with_swad
# lambda=1只是显式列出的候选：先做source-only pilot，不输出target分数
python make_jobs.py --dataset core6 --target 77GHz --recipes proposed --fishr --swad \
  --fishr-lambda 1 --source-only --seeds 42 --out plans/core6_fishr_pilot_lambda1
```

开启SWAD的每条轨迹都保留final-EMA和SWAD；固定Fishr超参数后，两组三种子训练可得到四格对照。Fishr必须明确传入`--fishr-lambda`，不自动沿用1000。lambda1000、anneal1500只是旧模型参考设置，不保证适合这个余弦头。预先列定候选，用`--source-only`完成源域验证，不查看target挑超参数。

source-only运行生成`SOURCE_ONLY_DONE.json`与`source_validation_final.json`。确定超参数、种子集合和选择器后，可在GPU作业内显式评测已经锁定的模型：

```bash
python evaluate_saved.py --run /完整的source_only_run路径 --selector final_ema \
  --settings-locked --out /新的评测输出目录
```

此命令必须在已分配GPU节点上执行。它检查选择器和checkpoint SHA256未被修改；不会根据target分数替你选择模型。普通GPU作业只需把训练命令替换为此命令，保留现有资源/环境设置。

`CosineHeadFishr`保留两头CE结构和梯度图，通过解析逐样本分类梯度实现，不要求向现有xcarrier环境安装BackPACK。标为适配版本，不冒称未改动官方Fishr。

## 结果文件与聚合

每个完整新run：

```
effective_config.json       # 有效参数，不是默认配置猜测
run_manifest.json           # protocol / versions / precision / seed
parameter_counts.json       # 三种参数量口径
model_structure.json        # 真实LoRA挂载位置及参数形状
audited_objective_used.py   # 实际执行的原forward/loss块
history.json                # 只含源域验证，target_evaluated=false
checkpoints/final_ema.pt
checkpoints/source_best_ema.pt
checkpoints/swad.pt         # 如开启SWAD
selection_frozen_before_target.json
predictions/*.npz           # 每图logits、标签、完整图片身份
predictions/*.csv
predictions/*.json
DONE.json
```

将同一任务/配置的三个完整run路径传入：

```bash
python aggregate.py --runs /完整run1 /完整run2 /完整run3 \
  --selector final_ema --out summaries/该配置
```

聚合会检查样本、标签、类顺序、seed与训练配置一致；不允许smoke混入，不根据target分数挑checkpoint。

配对消融比较：

```bash
python compare.py --treatment summaries/处理组 --control summaries/对照组 \
  --out summaries/该差异.json
```

统计区间是class-stratified image bootstrap，条件于这些已经训练好的模型；不能消除受试者/记录相关性或复制图片造成的依赖。`bootstrap_fraction_positive`不是常规假设检验p值。

## 实现与测试边界

- 原前向/损失块通过AST从原trainer提取执行；训练控制流程重新实现，目标域直到训练与选择冻结后才解码。
- 保留原采样、DAS/HFT/SpecAugment、两头训练分类、旧EMA初始化行为；辅助视图即使对应loss权重0仍由原代码构造，以免偷偷改变这部分执行路径。
- 改动控制流程、验证频率/批次或精度可能改变随机轨迹和数值，不能保证与旧训练逐位相同。
- 新模型/公式在小型编码器上的CPU测试通过；真实GPU训练仍待验证。
- 不支持resume；不擅自修复原resume遗漏状态的问题。
- 外部Table I baseline源码/预训练权重与full-FT target-best比较的缺口见AUDIT_REPORT。

许可证：用户提供的CrossCarrier源文件保留其原许可证；SWAD派生文件采用随附MIT许可，见vendor/SWAD_LICENSE。
