# 在 UBC Sockeye 运行复现与三方向训练

这套脚本已经准备好用于 Sockeye，但尚未登录你的 Sockeye，也没有在集群上运行训练。需要填入你实际的 allocation account、GPU partition、项目路径和环境激活脚本。脚本没有预设未经核实的 SSH 主机、分区名称或 module 版本。

## 先区分两件事

| 工作 | 运行内容 | 如何解释结果 |
|---|---|---|
| `repro` | 用学弟提供的三份最终 EMA checkpoint 重新评估 `10+24 → 77 GHz` | 这是已有权重的推理复现，不是重新训练；FP16/V100 与原硬件可能有数值差异 |
| `train` | 分别从头训练 `10+24 → 77`、`24+77 → 10`、`10+77 → 24`，每个方向三个 seed | 默认使用新的 grouped protocol；先修复原 split 的同源增强样本泄漏，再比较三个方向 |

新 grouped protocol 的 `→77` 训练结果不能直接当作原论文数字的严格重跑：数据划分已经发生变化。应把原 checkpoint 评估结果与新 protocol 的三方向结果分开报告。训练阶段不应根据目标频率的标签、F1 或 accuracy 选择 epoch/超参数。

grouped 划分保证 source train/validation 之间按原始记录分组隔离；它仍保留原任务的 known-person、同步记录跨频率设置，不能据此声称跨未见记录、未见 session 或未见受试者的泛化能力。

## 1. 登录并确认你的资源

在校外先连接 UBC VPN。Windows PowerShell 中使用开户邮件或 UBC Quickstart 给出的**真实登录主机**，由你完成 CWL/MFA：

```powershell
ssh YOUR_CWL@YOUR_SOCKEYE_LOGIN_HOST
```

Sockeye 于 2023 年 10 月 31 日改用 Slurm。GPU 为 V100 16GB/32GB，故本配置默认使用 `fp16`，训练代码配合 GradScaler。不要照搬原 RTX 5090 的 CUDA/PyTorch 环境，也不要强制 V100 使用 BF16。[Slurm 迁移说明](https://arc.ubc.ca/compute-storage/ubc-arc-sockeye/slurm)、[硬件规格](https://arc.ubc.ca/sockeye-techspecs)

登录节点用于文件管理、查看资源和提交任务。GPU 检查、评估和训练都通过本目录的 Slurm 脚本在计算节点运行。`gpu_job.sbatch` 和 `preflight.py` 会拒绝在缺少 `SLURM_JOB_ID` 时运行。

将代码上传后，从项目根目录执行以下只读检查：

```bash
bash hpc/probe.sh
```

重点查看 `sinfo` 的 GPU 资源/分区/最长时间、`scontrol show partition` 和自己可使用的 allocation account。`sacctmgr` 若被限制，可查看开户说明、ARC Access Management Portal 或询问 allocation 管理者。`module avail` 用来选择实际可用的 Python/环境管理模块。

详细集群文档需要 UBC 登录，本次无法公开读取确切的主机名、路径前缀、分区、walltime 上限及 module 版本。因此示例中的资源数值只是初始请求，必须按你的分区实际限制调整。[UBC Quickstart](https://confluence.it.ubc.ca/display/UARC/Quickstart+Guide)、[Technical User Documentation](https://confluence.it.ubc.ca/spaces/UARC/pages/168841652/UBC%2BARC%2BTechnical%2BUser%2BDocumentation)

## 2. 上传项目、数据和离线 backbone

建议把整个修改后的 `CrossCarrier_Repro` 文件夹或压缩包传到 allocation 的 project 空间，再在 Linux 解压。必须保留 `weights/`、`tasks/` 和原始 `EXPERIMENTSRESULT/`，只上传 `.py` 文件无法完成已有 checkpoint 复现。路径中不要只保留 Windows 盘符。

较大文件推荐通过 Globus：Windows 安装 Globus Connect Personal，将本机设为 endpoint，再选择自己的 Sockeye collection 和 project 目录。也可使用学校允许的 SCP/SFTP/数据传输节点，具体主机按文档填写。[UBC Globus 说明](https://arc.ubc.ca/software/globus)

目录安排建议：

- `PROJECT_DIR`：上传的完整项目与原始数据，使用你 allocation 的 project 路径。
- `OUTPUT_ROOT`：训练输出放可写 project 目录，或 active computation 用的 scratch 目录。
- Python 环境：放 allocation 允许的软件安装位置，通过 `ENV_SETUP` 激活。
- scratch 中的最终 checkpoint、manifest、配置、日志和结果表要及时复制到需要保留的 project/其他存储；scratch 可能被清理。

实际容量与路径以自己的 allocation 为准。当前公开 FAQ 和较旧存储页面的 project 配额存在差异，不能直接依赖网页中的固定容量。project 用于研究数据，scratch 用于计算中间文件，home 适合配置和小脚本。[官方 FAQ](https://arc.ubc.ca/sockeye/faq)

## 3. 准备 Linux Python 环境

先阅读根目录 `requirements-portable.txt`，按其中说明安装兼容 V100 和集群驱动的 PyTorch/torchvision 配对，再安装其余依赖。原 `requirements.txt` 记录的是 Windows/RTX 5090 环境，不能直接把 Windows venv 或 Conda 环境复制到 Sockeye。

你的环境激活脚本可以形如下面内容，保存到 project 中一个绝对路径；module 名和环境路径必须换成实际值：

```bash
#!/usr/bin/env bash
module load YOUR_ACTUAL_PYTHON_MODULE
source /YOUR_ACTUAL_ENV_PATH/bin/activate
```

如果你实际使用 Conda，激活脚本应 source 对应的 `conda.sh` 后 `conda activate /actual/env/path`，不要在 batch shell 中依赖交互式 `.bashrc` 自动激活。

例如，在已加载兼容 Python 且允许安装的 Linux 环境中创建 venv：

```bash
python -m venv /YOUR_ACTUAL_ENV_PATH
source /YOUR_ACTUAL_ENV_PATH/bin/activate
# 按 requirements-portable.txt 说明先安装 PyTorch/torchvision
python -m pip install -r requirements-portable.txt
```

一个具体的 **V100 候选环境（尚未在 Sockeye 测试）** 是 Linux x86_64、Python 3.10/3.11、`torch 2.7.1+cu118`、`torchvision 0.22.1+cu118`。PyTorch 官方给出了这对 CUDA 11.8 wheel 的安装命令；2.7.1 的官方构建脚本在 CUDA 11.8 分支包含 `sm_70`。先检查集群 Linux 的 `getconf GNU_LIBC_VERSION`：这些 wheel 使用 manylinux 2.28，需要 glibc 至少 2.28。如果宿主系统更旧，应使用 ARC 支持的兼容 module/容器方案，再验证本项目依赖，不能通过替换系统 glibc 解决。[官方安装配对](https://pytorch.org/get-started/previous-versions/#v271)、[2.7.1 构建脚本](https://github.com/pytorch/pytorch/blob/v2.7.1/.ci/manywheel/build_cuda.sh)、[manylinux 平台变更](https://dev-discuss.pytorch.org/t/pytorch-linux-wheels-switching-to-new-wheel-build-platform-manylinux-2-28-on-november-12-2024/2581)

确认 GPU 节点的实际驱动，可申请一个短 Slurm 作业执行下面命令；account、partition 和时限按自己的配置填写：

```bash
srun --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION \
  --nodes=1 --ntasks=1 --gres=gpu:1 --time=00:05:00 \
  nvidia-smi --query-gpu=name,driver_version --format=csv
```

CUDA 11.8 的 Linux toolkit 对应驱动为 `>=520.61.05`，这是本候选较稳妥的起点；NVIDIA 也给出了 CUDA 11.x minor compatibility 的最低 `450.80.02`，但兼容模式有功能限制，不能仅凭该较低阈值保证本模型正常运行。驱动更旧时先由 ARC 确认可用环境。后面的 GPU preflight 会记录真实 driver，并执行 CUDA 运算；再运行模型 smoke test 才能判断整套依赖是否工作。[CUDA 11.8 发布说明](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-toolkit-release-notes/)、[兼容模式限制](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)

在符合上述条件、允许联网安装的 Linux 环境中，候选安装命令为：

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements-portable.txt -c hpc/constraints-cu118.txt
python -m pip check
```

constraints 防止安装其余依赖时无意升级这对 torch/torchvision。如果安装冲突，应保留完整错误并重新选择兼容环境，不要直接删掉 pin 后将一个未经记录的新版本当作同一复现环境。

**不要在 V100 上照搬本机已测的 `torch 2.11.0+cu128`。** 官方 2.11 CUDA 12.8 wheel 已移除 Volta/V100，最低是 Turing；2.11 的 CUDA 12.6 wheel 仍列出 Volta 支持，但它需要另外核验集群驱动及整套依赖。选择环境时要同时检查 driver、Linux ABI 和 wheel 的 GPU 架构支持。[PyTorch 2.11 官方架构矩阵](https://dev-discuss.pytorch.org/t/dropping-volta-support-from-cuda-12-8-binaries-for-release-2-11/3290)

Sockeye 节点默认拒绝出站连接，仅特定节点允许有限外连。应在 ARC 允许的环境完成依赖安装；不能假设 GPU 作业中可以运行 `pip` 或下载模型。若需离线安装，准备**Linux x86_64、相同 Python ABI** 的 wheelhouse 后上传，按 requirements 使用 `pip --no-index --find-links` 安装；不要使用 Windows wheel。该包已有 DINOv3 backbone，任务脚本会强制 Hugging Face/Transformers 离线，并检查缓存权重是否存在。[网络说明](https://arc.ubc.ca/security-privacy/arc-sockeye-security-and-privacy)、[软件安装说明](https://arc.ubc.ca/sockeye/software)

## 4. 填配置并准备 manifest

在项目根目录：

```bash
cp hpc/config.example.sh hpc/config.sh
```

编辑 `hpc/config.sh`，替换全部 `REPLACE_ME`。所有路径使用 Linux 绝对路径。`ENV_SETUP` 指向上一步的环境激活脚本。不需要把密码、CWL 密码或 token 放进配置文件。

配置中的 `32G` 内存、4 个 CPU、训练 6 小时是可调整的初始请求，不是已测量的训练需求，也不是集群最大限制。`TRAIN_BATCH_SIZE=16` 保持原训练 batch；如果 V100 16GB 显存不足，可降到 8，并在所有比较方向保持一致、记录该训练配方变化。`REPRO_BATCH_SIZE=8` 和 `EVAL_BATCH_SIZE=8` 只改变评估批量大小。`NUM_WORKERS` 不要超过申请的 CPU 数。

三方向训练需要根目录提供的 `prepare_cross_frequency.py` 生成 grouped manifests。该脚本只需要 Python 标准库，执行：

```bash
python prepare_cross_frequency.py --protocol grouped
```

默认固定划分 seed 为 42，source validation 比例为 20%。它会核对图片路径/哈希，并输出各方向 `summary.json` 和总审计 `tasks/cross_frequency_audit.json`。确认三个方向都有：

```text
tasks/cross_frequency_grouped/target77/manifest/
tasks/cross_frequency_grouped/target10/manifest/
tasks/cross_frequency_grouped/target24/manifest/
```

应只准备一次划分，并让所有 seed 使用相同 manifest；不要为每个 seed 重新随机划分数据。训练脚本显式传 `--protocol grouped`。

## 5. 先提交 GPU 环境检查和已有模型复现

```bash
bash hpc/submit.sh preflight
squeue -u "$USER"
```

查看 `OUTPUT_ROOT/slurm/` 和 `OUTPUT_ROOT/preflight/job-JOBID/console.log`。成功后会生成 `gpu_preflight.json`，记录 GPU、CUDA、PyTorch 及主要依赖版本；还会实际执行一次 FP16 前向/反向和 GradScaler 更新。**检查通过仅代表环境正常，不代表模型已复现。**

环境通过后，提交原方向 checkpoint 评估：

```bash
bash hpc/submit.sh repro
```

输出写到 `OUTPUT_ROOT/repro/job-JOBID/`，不会覆盖项目附带的历史结果。每个作业保存 `console.log`、`environment.freeze.txt` 和 GPU 环境报告；`reproduce.py` 保存自己的评估输出。将实测 macro-F1、每 seed 指标和 ensemble 指标同学弟提供的数值逐项比较。换 GPU/精度后的微小数值差异需要记录，不应把脚本打印的 expected 值当作本次测得结果。

## 6. 提交三个方向、三个 seed 的训练

先做短训练检查：复制一个配置文件，将其中 `EPOCHS=1`、`TRAIN_MAX_STEPS=8`，其他参数不变。独立配置文件防止修改尚在排队的正式任务：

```bash
cp hpc/config.sh hpc/config.smoke.sh
# 编辑 config.smoke.sh：EPOCHS=1，TRAIN_MAX_STEPS=8
bash hpc/submit.sh train hpc/config.smoke.sh
```

它会对三个方向和三个 seed 各运行最多 8 个训练 step，并在日志中标明 SMOKE TEST。FP16 的初始 GradScaler 可能在前几步遇到梯度溢出并跳过更新；8 步让它有机会自动降低缩放比例。日志会记录有效更新数，完全没有有效更新时脚本会明确失败。这些结果只能检查训练链路，不可作为识别精度结果。根据实际完整 epoch 耗时和内存，设置足够的正式 walltime；可另用 `EPOCHS=5`、`TRAIN_MAX_STEPS=''` 做完整 epoch 的计时。正常训练要求至少 5 个 epoch，因为原方法从第 5 个 epoch 才启动 EMA；这仍是计时实验，不是正式 100 epoch 结果。

短测试通过后，正式配置应为 `EPOCHS=100`、`TRAIN_MAX_STEPS=''`：

```bash
bash hpc/submit.sh train hpc/config.sh
```

脚本提交 `--array=0-8%1`，每个 array task 申请一张 GPU，默认最多同时运行一个。可在确认 allocation 资源允许后调高 `MAX_CONCURRENT`，范围 1–9。

| Array task | 源频率 → 目标频率 | Seed |
|---|---|---|
| 0 / 1 / 2 | 10+24 → 77 | 42 / 1234 / 31415 |
| 3 / 4 / 5 | 24+77 → 10 | 42 / 1234 / 31415 |
| 6 / 7 / 8 | 10+77 → 24 | 42 / 1234 / 31415 |

训练输出分别写入：

```text
OUTPUT_ROOT/train/job-ARRAYJOBID/proposed/target77/seed42/
OUTPUT_ROOT/train/job-ARRAYJOBID/proposed/target10/seed42/
OUTPUT_ROOT/train/job-ARRAYJOBID/proposed/target24/seed42/
```

另外两个 seed 对应各自的子目录。各 `seed` 目录保存 HPC 日志和环境元数据；其下的 `model/` 才是传给训练入口的 run-dir，保存模型/checkpoint/报告，避免训练入口把已有日志误认为混入旧实验。完整实验应在三个方向中使用同一 protocol、epochs、学习率、batch size 和评估流程，并同时报告每 seed 值与三 seed 的均值/标准差。

为了判断方法是否提高识别精度，应在相同 grouped 划分下运行对照。默认 `VARIANT='proposed'`；复制正式配置为 `hpc/config.baseline.sh`，仅将 `VARIANT='no_das_acr'` 后，再提交一组相同的 9 个任务：

```bash
bash hpc/submit.sh train hpc/config.baseline.sh
```

`no_das_acr` 保持相同 backbone、HFT、SpecAug 和优化器，关闭 DAS 和 GRL/频率残差损失；这是对应的消融对照，不代表所有可能的 source-only 基线。与 proposed 比较时要确认两个配置的 batch、epochs、precision 和 split 完全一致。对照的目录名为 `no_das_acr`。

任务提交后不要修改对应配置和代码，直到它们完成。当前训练入口没有承诺严格断点续训；被 walltime 中止后不能把重新执行当作完整、连续的 100 epoch 训练。应申请足够时限，或先为需要的续训行为补齐并验证模型、EMA、优化器、scheduler、对抗头、GradScaler 和随机状态的恢复。

## 7. 查看运行状态

```bash
squeue -u "$USER"
sacct -j YOUR_JOB_ID --format=JobID,State,ExitCode,Elapsed,MaxRSS,AllocTRES
tail -n 80 /YOUR_OUTPUT_ROOT/slurm/YOUR_LOG_FILE.out
```

`COMPLETED` 且 `ExitCode=0:0` 只是进程正常退出，还需要检查输出中的 epoch 数、样本数、protocol、频率方向及最终指标。`OUT_OF_MEMORY` 时分别检查 CPU 内存和 CUDA 显存；CUDA OOM 可以先降 batch。`TIMEOUT` 需要更合理的时限或可靠续训实现。环境预检能发现 Python/CUDA 构建不兼容，但不能代替真实模型短训练检查。

需要停止自己提交的作业时，可以执行 `scancel YOUR_JOB_ID`。重要训练产物应从 scratch 及时转存；Sockeye 不提供用户数据备份服务。[官方安全/备份说明](https://arc.ubc.ca/security-privacy/arc-sockeye-security-and-privacy)
