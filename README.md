# Qwen3-8B Base APRIL 4+4 no early discard + DAPO 训练进度

更新时间：2026-11-02T00:53:55+08:00（北京时间，计划每 10 分钟同步；网络失败时顺延）

方法：**APRIL 4+4 不提前舍弃 + DAPO**；独立实验，从相同 Base 重新开始，不接续上一组权重。每题先生成前 4 条，全部完成并判分后，不论对错都放行后 4 条；每个正常结束组完整生成 8 条回答。完整 G8 仍按原 DAPO 规则过滤全对或全错组，并非所有生成组都用于训练。相对上一轮 4+4，仅关闭提前舍弃，保留两段门控与其余配置。

**8 张 GPU · pending 窗口 W=256 组 · 每次更新 64 个完整有效组（512 条回答）· PPO mini-batch 16 组**。ready 池独立，不占 pending 窗口；完整有效或完整无效组退出 pending 后都补新题；不因前 4 条全对提前结束组。未完成回答可跨权重续写。

保留 CPU 状态处理优化；启动时做初始评估，之后每 **50 轮评估**、每 **50 轮保存检查点**，保留最近 2 个。10 小时从真实 GPU 分配开始计时，100,000 步仅为保护上限。

累计用时 **0.18 / 10 小时**；剩余 **9.82 小时**。包含初始化、生成、CPU 判分、验证、保存及收尾，不补回空档。

Qwen3-8B-Base · DeepMath open-v4 train 40,299 · 回答上限 8192 tokens · 学习率 5e-7。

**训练中 · 已完成 11 步 · 10 小时硬上限**

当前第 12 轮，阶段：行为概率计算。
计划截止：**2026-11-02T10:42:58+08:00**。

最近 10 轮平均耗时：**31.29 秒/轮**（含各轮验证；不含启动及停机空档）。
最新检查点：尚无已完成检查点标记。

## 验证集

[上一轮 4+4 提前舍弃完整归档](history/20261101-setup2-qwen3-8b-april-discard44-w256-g64-8k-from0-10h/README.md)：约 9 小时 58 分钟结束，末尾评估收尾异常，原始失败状态保留。完整轮次 502、已保存第 503 轮检查点、最新完整评估第 500 轮；不并入本轮曲线。

DeepMath hard-dev 固定 **1,021 题**；公共测评保留 AIME24 / AIME25 / AMC23 / MATH500 全部 **600 题**。每题采样 4 次，报告 **mean@4**（4 个回答正确率的均值），不是 pass@4；unknown 保留在分母并计错。

最新验证：第 **0** 步。

| 验证分组 | 题数 × 回答数 | mean@4 |
|---|---:|---:|
| DeepMath hard-dev | 1021 × 4 | 54.3340% |
| DeepMath D5-6 | 128 × 4 | 55.0781% |
| DeepMath D6-7 | 255 × 4 | 54.7059% |
| DeepMath D7-8 | 319 × 4 | 53.9969% |
| DeepMath D8-9 | 272 × 4 | 54.2279% |
| DeepMath D9-10 | 47 × 4 | 53.1915% |
| AIME24 | 30 × 4 | 13.3333% |
| AIME25 | 30 × 4 | 8.3333% |
| AMC23 | 40 × 4 | 53.1250% |
| MATH500 | 500 × 4 | 75.4500% |

![验证 mean@4](validation.svg?v=2a837356104a)

两个面板分别展示 DeepMath hard-dev（总体与五个难度桶）和四个公共测评集。hard-dev 总体按各桶真实题数加权；公共测评不合并成一个总分。仅连接实测检查点，无平滑。

<details>
<summary>各项历史最佳 mean@4</summary>

| 验证分组 | 最佳步数 | 历史最佳 mean@4 |
|---|---:|---:|
| DeepMath hard-dev | 0 | 54.3340% |
| DeepMath D5-6 | 0 | 55.0781% |
| DeepMath D6-7 | 0 | 54.7059% |
| DeepMath D7-8 | 0 | 53.9969% |
| DeepMath D8-9 | 0 | 54.2279% |
| DeepMath D9-10 | 0 | 53.1915% |
| AIME24 | 0 | 13.3333% |
| AIME25 | 0 | 8.3333% |
| AMC23 | 0 | 53.1250% |
| MATH500 | 0 | 75.4500% |

各项独立取最佳，不代表同一个检查点同时达到，也不代表该轮权重仍被保留。

</details>

## 最近十轮

| 轮次 | 总秒数（含验证） | 采集秒数 | 更新秒数 | 验证秒数 | 新候选组 / 完成组 | carry / 选中组 | pending / surplus |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 28.76 | 12.56 | 8.04 | — | 83 / 83 | 5 / 64 | 256 / 7 |
| 3 | 28.04 | 12.63 | 8.18 | — | 80 / 80 | 7 / 64 | 256 / 6 |
| 4 | 35.49 | 18.72 | 9.89 | — | 86 / 86 | 6 / 64 | 256 / 2 |
| 5 | 32.71 | 16.83 | 8.77 | — | 100 / 100 | 2 / 64 | 256 / 5 |
| 6 | 31.07 | 12.43 | 11.23 | — | 81 / 81 | 5 / 64 | 256 / 2 |
| 7 | 31.59 | 15.11 | 9.50 | — | 88 / 88 | 2 / 64 | 256 / 8 |
| 8 | 31.80 | 13.69 | 10.84 | — | 86 / 86 | 8 / 64 | 256 / 8 |
| 9 | 29.05 | 13.16 | 9.01 | — | 74 / 74 | 8 / 64 | 256 / 2 |
| 10 | 32.14 | 14.10 | 10.78 | — | 82 / 82 | 2 / 64 | 256 / 1 |
| 11 | 32.20 | 14.36 | 10.17 | — | 88 / 88 | 1 / 64 | 256 / 2 |

采集包含续写、CPU 判分与队列处理；carry 为采集前已完成的有效组，pending 为暂停后未完成组，surplus 为留到后轮的完整有效组。未记录的值显示‘—’，不填零；总秒数优先使用轮次结束事件，缺失时仅在两项计时均存在时使用 step + testing。

## GPU 0–7

| GPU | 利用率 | 已用 / 总显存 GiB | 最新采样（北京时间） | 采样状态 |
|---:|---:|---:|---|---|
| 0 | 97% | 25.73 / 268.59 | 2026-11-02T00:53:55+08:00 | ok |
| 1 | 97% | 26.40 / 268.59 | 2026-11-02T00:53:55+08:00 | ok |
| 2 | 97% | 26.65 / 268.59 | 2026-11-02T00:53:55+08:00 | ok |
| 3 | 88% | 26.42 / 268.59 | 2026-11-02T00:53:55+08:00 | ok |
| 4 | 65% | 26.52 / 268.59 | 2026-11-02T00:53:55+08:00 | ok |
| 5 | 74% | 26.69 / 268.59 | 2026-11-02T00:53:55+08:00 | ok |
| 6 | 97% | 26.15 / 268.59 | 2026-11-02T00:53:55+08:00 | ok |
| 7 | 97% | 26.69 / 268.59 | 2026-11-02T00:53:55+08:00 | ok |

最新值来源：1 秒滚动快照。数值均为瞬时采样，失败采样留空，不当成零。

## 最近十分钟 GPU 利用率

每秒采样；当前实际覆盖 **599.0 / 600 秒**，599 个采样时刻。
窗口：2026-11-02T00:43:56+08:00 至 2026-11-02T00:53:55+08:00。
快照距本页生成 **0.6 秒**（新鲜）。

![最近十分钟 GPU 利用率](gpu_utilization_10m.svg?v=fdc6054f115a)

![最近十分钟 GPU 显存](gpu_memory_10m.svg?v=53601ab83e81)

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
- [20261101-setup2-qwen3-8b-april-discard44-w256-g64-8k-from0-10h](history/20261101-setup2-qwen3-8b-april-discard44-w256-g64-8k-from0-10h/README.md)
- [20261102-setup2-qwen3-8b-april-nodiscard44-w256-g64-8k-from0-10h](history/20261102-setup2-qwen3-8b-april-nodiscard44-w256-g64-8k-from0-10h/README.md)
