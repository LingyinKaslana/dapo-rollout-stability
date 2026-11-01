# Qwen3-8B Base APRIL 4+4 no early discard + DAPO 训练进度

更新时间：2026-11-02T03:29:02+08:00（北京时间，计划每 10 分钟同步；网络失败时顺延）

方法：**APRIL 4+4 不提前舍弃 + DAPO**；独立实验，从相同 Base 重新开始，不接续上一组权重。每题先生成前 4 条，全部完成并判分后，不论对错都放行后 4 条；每个正常结束组完整生成 8 条回答。完整 G8 仍按原 DAPO 规则过滤全对或全错组，并非所有生成组都用于训练。相对上一轮 4+4，仅关闭提前舍弃，保留两段门控与其余配置。

**8 张 GPU · pending 窗口 W=256 组 · 每次更新 64 个完整有效组（512 条回答）· PPO mini-batch 16 组**。ready 池独立，不占 pending 窗口；完整有效或完整无效组退出 pending 后都补新题；不因前 4 条全对提前结束组。未完成回答可跨权重续写。

保留 CPU 状态处理优化；启动时做初始评估，之后每 **50 轮评估**、每 **50 轮保存检查点**，保留最近 2 个。10 小时从真实 GPU 分配开始计时，100,000 步仅为保护上限。

累计用时 **2.77 / 10 小时**；剩余 **7.23 小时**。包含初始化、生成、CPU 判分、验证、保存及收尾，不补回空档。

Qwen3-8B-Base · DeepMath open-v4 train 40,299 · 回答上限 8192 tokens · 学习率 5e-7。

**训练中 · 已完成 249 步 · 10 小时硬上限**

当前第 250 轮，阶段：生成 / 续写。
计划截止：**2026-11-02T10:42:58+08:00**。

最近 10 轮平均耗时：**90.10 秒/轮**（含各轮验证；不含启动及停机空档）。
最新检查点：第 **200** 轮；必需文件非空检查 **25/25**（仅文件存在性检查）。

## 验证集

[上一轮 4+4 提前舍弃完整归档](history/20261101-setup2-qwen3-8b-april-discard44-w256-g64-8k-from0-10h/README.md)：约 9 小时 58 分钟结束，末尾评估收尾异常，原始失败状态保留。完整轮次 502、已保存第 503 轮检查点、最新完整评估第 500 轮；不并入本轮曲线。

DeepMath hard-dev 固定 **1,021 题**；公共测评保留 AIME24 / AIME25 / AMC23 / MATH500 全部 **600 题**。每题采样 4 次，报告 **mean@4**（4 个回答正确率的均值），不是 pass@4；unknown 保留在分母并计错。

最新验证：第 **200** 步。

| 验证分组 | 题数 × 回答数 | mean@4 |
|---|---:|---:|
| DeepMath hard-dev | 1021 × 4 | 67.3604% |
| DeepMath D5-6 | 128 × 4 | 66.0156% |
| DeepMath D6-7 | 255 × 4 | 67.3529% |
| DeepMath D7-8 | 319 × 4 | 68.5737% |
| DeepMath D8-9 | 272 × 4 | 65.9926% |
| DeepMath D9-10 | 47 × 4 | 70.7447% |
| AIME24 | 30 × 4 | 16.6667% |
| AIME25 | 30 × 4 | 14.1667% |
| AMC23 | 40 × 4 | 65.0000% |
| MATH500 | 500 × 4 | 81.7000% |

![验证 mean@4](validation.svg?v=92df1b522f29)

两个面板分别展示 DeepMath hard-dev（总体与五个难度桶）和四个公共测评集。hard-dev 总体按各桶真实题数加权；公共测评不合并成一个总分。仅连接实测检查点，无平滑。

<details>
<summary>各项历史最佳 mean@4</summary>

| 验证分组 | 最佳步数 | 历史最佳 mean@4 |
|---|---:|---:|
| DeepMath hard-dev | 200 | 67.3604% |
| DeepMath D5-6 | 200 | 66.0156% |
| DeepMath D6-7 | 200 | 67.3529% |
| DeepMath D7-8 | 200 | 68.5737% |
| DeepMath D8-9 | 200 | 65.9926% |
| DeepMath D9-10 | 200 | 70.7447% |
| AIME24 | 200 | 16.6667% |
| AIME25 | 50 | 15.0000% |
| AMC23 | 100 | 68.1250% |
| MATH500 | 200 | 81.7000% |

各项独立取最佳，不代表同一个检查点同时达到，也不代表该轮权重仍被保留。

</details>

## 最近十轮

| 轮次 | 总秒数（含验证） | 采集秒数 | 更新秒数 | 验证秒数 | 新候选组 / 完成组 | carry / 选中组 | pending / surplus |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 240 | 86.72 | 50.07 | 25.18 | — | 124 / 124 | 21 / 64 | 256 / 19 |
| 241 | 92.56 | 56.99 | 25.10 | — | 128 / 128 | 19 / 64 | 256 / 20 |
| 242 | 85.89 | 51.21 | 23.64 | — | 105 / 105 | 20 / 64 | 256 / 11 |
| 243 | 93.24 | 55.41 | 26.04 | — | 113 / 113 | 11 / 64 | 256 / 16 |
| 244 | 93.40 | 55.40 | 26.68 | — | 124 / 124 | 16 / 64 | 256 / 21 |
| 245 | 85.35 | 47.91 | 25.91 | — | 106 / 106 | 21 / 64 | 256 / 20 |
| 246 | 93.67 | 55.91 | 26.42 | — | 122 / 122 | 20 / 64 | 256 / 23 |
| 247 | 85.57 | 51.00 | 23.08 | — | 108 / 108 | 23 / 64 | 256 / 18 |
| 248 | 97.23 | 61.45 | 24.79 | — | 131 / 131 | 18 / 64 | 256 / 22 |
| 249 | 87.34 | 48.81 | 26.86 | — | 112 / 112 | 22 / 64 | 256 / 15 |

采集包含续写、CPU 判分与队列处理；carry 为采集前已完成的有效组，pending 为暂停后未完成组，surplus 为留到后轮的完整有效组。未记录的值显示‘—’，不填零；总秒数优先使用轮次结束事件，缺失时仅在两项计时均存在时使用 step + testing。

## GPU 0–7

| GPU | 利用率 | 已用 / 总显存 GiB | 最新采样（北京时间） | 采样状态 |
|---:|---:|---:|---|---|
| 0 | 100% | 189.65 / 268.59 | 2026-11-02T03:29:01+08:00 | ok |
| 1 | 98% | 190.57 / 268.59 | 2026-11-02T03:29:01+08:00 | ok |
| 2 | 100% | 190.60 / 268.59 | 2026-11-02T03:29:01+08:00 | ok |
| 3 | 100% | 190.60 / 268.59 | 2026-11-02T03:29:01+08:00 | ok |
| 4 | 100% | 190.53 / 268.59 | 2026-11-02T03:29:01+08:00 | ok |
| 5 | 100% | 190.57 / 268.59 | 2026-11-02T03:29:01+08:00 | ok |
| 6 | 100% | 190.65 / 268.59 | 2026-11-02T03:29:01+08:00 | ok |
| 7 | 100% | 190.56 / 268.59 | 2026-11-02T03:29:01+08:00 | ok |

最新值来源：1 秒滚动快照。数值均为瞬时采样，失败采样留空，不当成零。

## 最近十分钟 GPU 利用率

每秒采样；当前实际覆盖 **599.0 / 600 秒**，599 个采样时刻。
窗口：2026-11-02T03:19:02+08:00 至 2026-11-02T03:29:01+08:00。
快照距本页生成 **0.8 秒**（新鲜）。

![最近十分钟 GPU 利用率](gpu_utilization_10m.svg?v=5a73ab979d9f)

![最近十分钟 GPU 显存](gpu_memory_10m.svg?v=457fa07bb848)

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
