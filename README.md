# Qwen3-8B Base APRIL 8-simultaneous race-four discard + DAPO 训练进度

更新时间：2026-11-02T13:27:25+08:00（北京时间，计划每 10 分钟同步；网络失败时顺延）

方法：**APRIL 8 同发、最先结束的 4 条全对则舍弃 + DAPO**；独立实验，从相同 Base 重新开始，不接续上一组权重。每题 8 条同时放行，按生成完成顺序确定最先结束的 4 条，等待这 4 条判分；若全对则舍弃整组，并取消本组其余仍在运行的请求，否则保留完整 8 条。完整 G8 仍按原 DAPO 规则过滤全对或全错组。取消前已经生成的 token 均计入成本，不假设固定节省 4 条回答。

**8 张 GPU · pending 窗口 W=256 组 · 每次更新 64 个完整有效组（512 条回答）· PPO mini-batch 16 组**。ready 池独立，不占 pending 窗口；完整有效、完整无效或已确认取消收尾的舍弃组退出 pending 后都补新题。未完成回答可跨权重续写。

保留 CPU 状态处理优化；启动时做初始评估，之后每 **50 轮评估**、每 **50 轮保存检查点**，保留最近 2 个。10 小时从真实 GPU 分配开始计时，100,000 步仅为保护上限。

累计用时 **0.59 / 10 小时**；剩余 **9.41 小时**。包含初始化、生成、CPU 判分、验证、保存及收尾，不补回空档。

Qwen3-8B-Base · DeepMath open-v4 train 40,299 · 回答上限 8192 tokens · 学习率 5e-7。

**训练中 · 已完成 49 步 · 10 小时硬上限**

当前第 50 轮，阶段：验证。
计划截止：**2026-11-02T22:52:14+08:00**。

最近 10 轮平均耗时：**34.87 秒/轮**（含各轮验证；不含启动及停机空档）。
最新检查点：第 **50** 轮；必需文件非空检查 **25/25**（仅文件存在性检查）。

## 抢先四条舍弃统计

累计舍弃 **1421** 组 / 已闭合 **5715** 组；舍弃率 **24.86%**。分母为舍弃组加完整 G8 结束组，不含尚未完成组，也不重复计入 ready 池。
本组剩余请求：实际取消 **0** 条，已生成结束（含取消竞态中自然结束） **5018** 条，未开始 **0** 条，跨权重暂停期间取消 **666** 条。
舍弃组已消耗 **9138256 tokens**，其中首四条 **3759934**、其余四条 **5378322**；这些 token 不进入训练。不能把舍弃一组换算成固定节省 4 条回答。
口径：截至第 **49** 轮完整指标；未完成当前轮的增量尚未纳入。缺测显示‘—’，不当作零。

## 验证集

[上一轮 4+4 不提前舍弃完整归档](history/20261102-setup2-qwen3-8b-april-nodiscard44-w256-g64-8k-from0-10h/README.md)：约 9 小时 57 分钟正常结束；完整轮次、最后检查点和最后完整评估均为 541。原始归档只读，不并入本轮曲线。

DeepMath hard-dev 固定 **1,021 题**；公共测评保留 AIME24 / AIME25 / AMC23 / MATH500 全部 **600 题**。每题采样 4 次，报告 **mean@4**（4 个回答正确率的均值），不是 pass@4；unknown 保留在分母并计错。

最新验证：第 **0** 步。

| 验证分组 | 题数 × 回答数 | mean@4 |
|---|---:|---:|
| DeepMath hard-dev | 1021 × 4 | 54.5788% |
| DeepMath D5-6 | 128 × 4 | 57.2266% |
| DeepMath D6-7 | 255 × 4 | 52.7451% |
| DeepMath D7-8 | 319 × 4 | 56.6614% |
| DeepMath D8-9 | 272 × 4 | 52.2059% |
| DeepMath D9-10 | 47 × 4 | 56.9149% |
| AIME24 | 30 × 4 | 13.3333% |
| AIME25 | 30 × 4 | 11.6667% |
| AMC23 | 40 × 4 | 51.2500% |
| MATH500 | 500 × 4 | 74.2500% |

![验证 mean@4](validation.svg?v=bc7476f95f6a)

两个面板分别展示 DeepMath hard-dev（总体与五个难度桶）和四个公共测评集。hard-dev 总体按各桶真实题数加权；公共测评不合并成一个总分。仅连接实测检查点，无平滑。

<details>
<summary>各项历史最佳 mean@4</summary>

| 验证分组 | 最佳步数 | 历史最佳 mean@4 |
|---|---:|---:|
| DeepMath hard-dev | 0 | 54.5788% |
| DeepMath D5-6 | 0 | 57.2266% |
| DeepMath D6-7 | 0 | 52.7451% |
| DeepMath D7-8 | 0 | 56.6614% |
| DeepMath D8-9 | 0 | 52.2059% |
| DeepMath D9-10 | 0 | 56.9149% |
| AIME24 | 0 | 13.3333% |
| AIME25 | 0 | 11.6667% |
| AMC23 | 0 | 51.2500% |
| MATH500 | 0 | 74.2500% |

各项独立取最佳，不代表同一个检查点同时达到，也不代表该轮权重仍被保留。

</details>

## 最近十轮

| 轮次 | 总秒数（含验证） | 采集秒数 | 更新秒数 | 验证秒数 | 新候选组 / 完成组 | carry / 选中组 | pending / surplus |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 40 | 16.49 | 0.27 | 8.89 | — | 0 / 0 | 72 / 64 | 256 / 8 |
| 41 | 47.59 | 31.30 | 8.96 | — | 235 / 166 | 8 / 64 | 256 / 67 |
| 42 | 17.66 | 0.27 | 9.93 | — | 0 / 0 | 67 / 64 | 256 / 3 |
| 43 | 48.99 | 33.62 | 8.02 | — | 238 / 162 | 3 / 64 | 256 / 60 |
| 44 | 43.54 | 25.51 | 9.97 | — | 158 / 101 | 60 / 64 | 256 / 70 |
| 45 | 16.71 | 0.27 | 9.59 | — | 0 / 0 | 70 / 64 | 256 / 6 |
| 46 | 50.49 | 33.79 | 8.66 | — | 243 / 159 | 6 / 64 | 256 / 64 |
| 47 | 15.98 | 0.25 | 8.33 | — | 0 / 0 | 64 / 64 | 256 / 0 |
| 48 | 50.38 | 32.84 | 10.35 | — | 277 / 172 | 0 / 64 | 256 / 62 |
| 49 | 40.89 | 25.25 | 8.31 | — | 145 / 90 | 62 / 64 | 256 / 70 |

采集包含续写、CPU 判分与队列处理；carry 为采集前已完成的有效组，pending 为暂停后未完成组，surplus 为留到后轮的完整有效组。未记录的值显示‘—’，不填零；总秒数优先使用轮次结束事件，缺失时仅在两项计时均存在时使用 step + testing。

## GPU 0–7

| GPU | 利用率 | 已用 / 总显存 GiB | 最新采样（北京时间） | 采样状态 |
|---:|---:|---:|---|---|
| 0 | 99% | 189.06 / 268.59 | 2026-11-02T13:27:25+08:00 | ok |
| 1 | 99% | 190.03 / 268.59 | 2026-11-02T13:27:25+08:00 | ok |
| 2 | 99% | 190.01 / 268.59 | 2026-11-02T13:27:25+08:00 | ok |
| 3 | 100% | 189.96 / 268.59 | 2026-11-02T13:27:25+08:00 | ok |
| 4 | 99% | 190.04 / 268.59 | 2026-11-02T13:27:25+08:00 | ok |
| 5 | 100% | 190.04 / 268.59 | 2026-11-02T13:27:25+08:00 | ok |
| 6 | 96% | 189.98 / 268.59 | 2026-11-02T13:27:25+08:00 | ok |
| 7 | 96% | 189.96 / 268.59 | 2026-11-02T13:27:25+08:00 | ok |

最新值来源：1 秒滚动快照。数值均为瞬时采样，失败采样留空，不当成零。

## 最近十分钟 GPU 利用率

每秒采样；当前实际覆盖 **599.0 / 600 秒**，600 个采样时刻。
窗口：2026-11-02T13:17:26+08:00 至 2026-11-02T13:27:25+08:00。
快照距本页生成 **0.0 秒**（新鲜）。

![最近十分钟 GPU 利用率](gpu_utilization_10m.svg?v=b506fcb25760)

![最近十分钟 GPU 显存](gpu_memory_10m.svg?v=34fc37a33b3c)

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
- [20261102-setup2-qwen3-8b-april-race4-w256-g64-8k-from0-10h](history/20261102-setup2-qwen3-8b-april-race4-w256-g64-8k-from0-10h/README.md)
