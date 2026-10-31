# Qwen3-8B Base APRIL full G8 concurrent + DAPO 训练进度

更新时间：2026-11-01T00:34:00+08:00（北京时间，每约 60 秒同步）

方法：**APRIL 普通 8 同发 + DAPO**；独立实验，从 Base 重新开始。每题一次提交全部 8 条回答，无 4+4 门控，不提前舍弃；完成完整 G8 后，按原 DAPO 规则过滤全对或全错组。

**8 张 GPU · pending 窗口 W=256 组 · 每次更新 64 个完整有效组（512 条回答）· PPO mini-batch 16 组**。ready 池独立，不占 pending 窗口；完整组退出 pending 后补新题。未完成回答可跨权重续写。

保留 CPU 状态处理优化；启动时做初始评估，之后每 **50 轮评估**、每 **50 轮保存检查点**，保留最近 2 个。12 小时从真实 GPU 分配开始计时，100,000 步仅为保护上限。

累计用时 **1.70 / 12 小时**；剩余 **10.30 小时**。包含初始化、生成、CPU 判分、验证、保存及收尾，不补回空档。

Qwen3-8B-Base · DeepMath open-v4 train 40,299 · 回答上限 8192 tokens · 学习率 5e-7。

**训练中 · 已完成 168 步 · 12 小时硬上限**

当前第 169 轮，阶段：权重同步。
计划截止：**2026-11-01T10:51:46+08:00**。

最近 10 轮平均耗时：**32.07 秒/轮**（含各轮验证；不含启动及停机空档）。
最新检查点：第 **150** 轮；必需文件非空检查 **25/25**（仅文件存在性检查）。

## 验证集

[上一轮独立实验完整归档](history/20261030-april-eight-w256-g64-eval50-from0-20h/README.md)。旧实验与本轮独立保存，不并入本轮曲线。

DeepMath hard-dev 固定 **1,021 题**；公共测评保留 AIME24 / AIME25 / AMC23 / MATH500 全部 **600 题**。每题采样 4 次，报告 **mean@4**（4 个回答正确率的均值），不是 pass@4；unknown 保留在分母并计错。

最新验证：第 **150** 步。

| 验证分组 | 题数 × 回答数 | mean@4 |
|---|---:|---:|
| DeepMath hard-dev | 1021 × 4 | 65.7444% |
| DeepMath D5-6 | 128 × 4 | 66.6016% |
| DeepMath D6-7 | 255 × 4 | 64.6078% |
| DeepMath D7-8 | 319 × 4 | 66.7712% |
| DeepMath D8-9 | 272 × 4 | 65.6250% |
| DeepMath D9-10 | 47 × 4 | 63.2979% |
| AIME24 | 30 × 4 | 16.6667% |
| AIME25 | 30 × 4 | 16.6667% |
| AMC23 | 40 × 4 | 60.0000% |
| MATH500 | 500 × 4 | 80.0500% |

![验证 mean@4](validation.svg?v=bdc99da26873)

两个面板分别展示 DeepMath hard-dev（总体与五个难度桶）和四个公共测评集。hard-dev 总体按各桶真实题数加权；公共测评不合并成一个总分。仅连接实测检查点，无平滑。

<details>
<summary>各项历史最佳 mean@4</summary>

| 验证分组 | 最佳步数 | 历史最佳 mean@4 |
|---|---:|---:|
| DeepMath hard-dev | 150 | 65.7444% |
| DeepMath D5-6 | 150 | 66.6016% |
| DeepMath D6-7 | 150 | 64.6078% |
| DeepMath D7-8 | 150 | 66.7712% |
| DeepMath D8-9 | 150 | 65.6250% |
| DeepMath D9-10 | 100 | 64.3617% |
| AIME24 | 150 | 16.6667% |
| AIME25 | 150 | 16.6667% |
| AMC23 | 100 | 62.5000% |
| MATH500 | 100 | 80.4500% |

各项独立取最佳，不代表同一个检查点同时达到，也不代表该轮权重仍被保留。

</details>

## 最近十轮

| 轮次 | 总秒数（含验证） | 采集秒数 | 更新秒数 | 验证秒数 | 新候选组 / 完成组 | carry / 选中组 | pending / surplus |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 159 | 45.89 | 29.41 | 8.49 | — | 252 / 252 | 23 / 64 | 256 / 81 |
| 160 | 16.34 | 0.26 | 9.17 | — | 0 / 0 | 81 / 64 | 256 / 17 |
| 161 | 50.68 | 33.38 | 9.18 | — | 249 / 249 | 17 / 64 | 256 / 84 |
| 162 | 15.92 | 0.29 | 9.05 | — | 0 / 0 | 84 / 64 | 256 / 20 |
| 163 | 47.81 | 30.64 | 9.39 | — | 236 / 236 | 20 / 64 | 256 / 74 |
| 164 | 16.26 | 0.25 | 8.65 | — | 0 / 0 | 74 / 64 | 256 / 10 |
| 165 | 49.01 | 31.73 | 9.49 | — | 261 / 261 | 10 / 64 | 256 / 88 |
| 166 | 16.50 | 0.26 | 9.52 | — | 0 / 0 | 88 / 64 | 256 / 24 |
| 167 | 47.29 | 31.84 | 7.73 | — | 245 / 245 | 24 / 64 | 256 / 88 |
| 168 | 14.99 | 0.25 | 7.85 | — | 0 / 0 | 88 / 64 | 256 / 24 |

采集包含续写、CPU 判分与队列处理；carry 为采集前已完成的有效组，pending 为暂停后未完成组，surplus 为留到后轮的完整有效组。未记录的值显示‘—’，不填零；总秒数优先使用轮次结束事件，缺失时仅在两项计时均存在时使用 step + testing。

## GPU 0–7

| GPU | 利用率 | 已用 / 总显存 GiB | 最新采样（北京时间） | 采样状态 |
|---:|---:|---:|---|---|
| 0 | 98% | 39.39 / 268.59 | 2026-11-01T00:33:59+08:00 | ok |
| 1 | 94% | 40.31 / 268.59 | 2026-11-01T00:33:59+08:00 | ok |
| 2 | 92% | 40.37 / 268.59 | 2026-11-01T00:33:59+08:00 | ok |
| 3 | 93% | 41.19 / 268.59 | 2026-11-01T00:33:59+08:00 | ok |
| 4 | 90% | 41.11 / 268.59 | 2026-11-01T00:33:59+08:00 | ok |
| 5 | 92% | 41.15 / 268.59 | 2026-11-01T00:33:59+08:00 | ok |
| 6 | 92% | 41.30 / 268.59 | 2026-11-01T00:33:59+08:00 | ok |
| 7 | 90% | 40.39 / 268.59 | 2026-11-01T00:33:59+08:00 | ok |

最新值来源：1 秒滚动快照。数值均为瞬时采样，失败采样留空，不当成零。

## 最近十分钟 GPU 利用率

每秒采样；当前实际覆盖 **599.0 / 600 秒**，599 个采样时刻。
窗口：2026-11-01T00:24:00+08:00 至 2026-11-01T00:33:59+08:00。
快照距本页生成 **0.8 秒**（新鲜）。

![最近十分钟 GPU 利用率](gpu_utilization_10m.svg?v=330d7aa8f072)

![最近十分钟 GPU 显存](gpu_memory_10m.svg?v=db22eac1c1dc)

8 卡逐行展示、共享北京时间轴，利用率面板固定 0–100% 范围和颜色；无平滑，失败采样与缺口处断线。刚启动不足十分钟时仅展示已有样本，分钟长历史单独保留。

[状态 JSON](status.json) · [验证 CSV](validation.csv) · [训练耗时 CSV](training.csv) · [GPU 最新值](gpu_latest.csv) · [GPU 秒级窗口](gpu_10m.csv) · [GPU 分钟历史](gpu_history.csv) · [验证矢量图](validation.svg) · [GPU 矢量图](gpu_utilization_10m.svg)

<details>
<summary>数据口径与发布范围</summary>

独立校准集 508 题不参与训练。训练池仍保留 1,629 条 review-only 题，规则筛查不代表所有题意或金标已被人工证明无误。
只发布汇总数值和图，不发布题目、回答、CoT、题目 ID、权重或私有路径。

</details>

## 历史实验

- [20261021-dapo-deepmath7-scalar-400](history/20261021-dapo-deepmath7-scalar-400/README.md)
- [20261022-base-deepmath-20k-24h](history/20261022-base-deepmath-20k-24h/README.md)
- [20261023-april-deepmath-20k-24h](history/20261023-april-deepmath-20k-24h/README.md)
- [20261025-rollpacker-deepmath-20k-24h](history/20261025-rollpacker-deepmath-20k-24h/README.md)
- [20261025-rollpacker-deepmath-70-fresh-24h](history/20261025-rollpacker-deepmath-70-fresh-24h/README.md)
- [20261026-tailsieve-deepmath-64-fresh-24h](history/20261026-tailsieve-deepmath-64-fresh-24h/README.md)
- [20261027-rollverify-deepmath-112-fresh-24h](history/20261027-rollverify-deepmath-112-fresh-24h/README.md)
- [20261029-april-discard8-w256-g64-from0-24h](history/20261029-april-discard8-w256-g64-from0-24h/README.md)
- [20261030-april-cpuopt-discard8-w256-g64-eval50-from0-24h](history/20261030-april-cpuopt-discard8-w256-g64-eval50-from0-24h/README.md)
- [20261030-april-cpuopt-discard8-w256-g64-from0-24h](history/20261030-april-cpuopt-discard8-w256-g64-from0-24h/README.md)
- [20261030-april-eight-w256-g64-eval50-from0-20h](history/20261030-april-eight-w256-g64-eval50-from0-20h/README.md)
- [20261031-setup2-qwen3-8b-april-eight-w256-g64-8k-from0-12h](history/20261031-setup2-qwen3-8b-april-eight-w256-g64-8k-from0-12h/README.md)
