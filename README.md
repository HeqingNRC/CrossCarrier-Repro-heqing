# CrossCarrier-Repro-heqing

Radar activity recognition across 10, 24 and 77 GHz, based on Zirui Lin's CrossCarrier reproduction code.

## 三个训练方向

- 10 + 24 → 77 GHz
- 10 + 77 → 24 GHz
- 24 + 77 → 10 GHz

源码在 [CrossCarrier/](CrossCarrier/)；运行命令均在该目录执行。

- [完整运行说明与验证状态](CrossCarrier/README.md)
- [UBC Sockeye 环境与 Slurm 指南](CrossCarrier/hpc/README_zh.md)
- [数据来源说明](CrossCarrier/docs/DATA_SOURCES.md)
- [原作者代码许可证](CrossCarrier/LICENSE)

## 在 Sockeye 获取代码

```bash
cd /scratch/st-zliu-1/heqingz
git clone https://github.com/HeqingNRC/CrossCarrier-Repro-heqing.git
cd CrossCarrier-Repro-heqing/CrossCarrier
```

私有仓库需要自己的 GitHub 认证。随后按完整运行说明接入已上传到 Sockeye 的数据与权重，并根据 `bash hpc/probe.sh` 的输出填写个人 HPC 配置。

仓库只包含代码、模板和文档。图片、权重、checkpoint、运行结果和个人 HPC 配置保留在本地或 Sockeye。

## 验证状态

已有 checkpoint 的本地推理复现和三个方向的短程 GPU 检查已完成；完整资产下 30 项测试通过，纯代码包 27 项通过、3 项数据集成测试跳过。
正式 100 epoch 三方向训练尚未完成，尚未在 Sockeye 提交作业。
