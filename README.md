# Qwen3-8B Base APRIL sequential-first-success + DAPO 训练进度

更新时间：2026-11-03T07:34:03+00:00（UTC，计划每 10 分钟同步；网络失败时顺延）

方法：同一题每次只生成 **1 条独立回答**，判分后若错误则继续；首次成功或最多 8 次后停止。首条就正确的 G1 组与 8 条全错组不训练；有效组为 **×…×√，G=2–8**。后一次仍输入原题，不把前次错误回答拼进 prompt。

**8 张 GPU · pending 窗口 W=512 组 · 每次更新恰好 512 条真实回答 · 有效组数可变**。同一题固定路由到同一张 GPU，沿用同权重的原生前缀缓存；不复用跨权重 KV。并行的是不同题，每题最多一条回答在途。

按完整有效组装箱；不截组、不重复回答、不补假奖励。ready 池独立，剩余完整组留到后续。原生 optimizer 仍为 **4 × 128 条回答**；配置中的 nominal n=8 / train_batch=64 / mini_batch=16 只是训练几何，不代表每轮选中 64 组或每题实际生成 8 条。

独立实验：从同一个 Qwen3-8B-Base 重新开始，全参数训练。总上限 **10 小时（80 GPUh）**，包含初始化、训练、测评、保存与收尾；**9 小时 45 分开始边界收尾**，不追加预算。

DeepMath open-v4 train 40,299 · 回答上限 8192 tokens · 学习率 5e-7。启动时初始评估，之后每 **50 轮评估**、每 **50 轮保存**，保留最近 2 个检查点。

**训练中 · 已记录 139 轮完整训练指标 · 实际分配 2.58 小时**

当前轮次：140；阶段：生成 / 续写。
收尾时间：**2026-11-03T14:44:12+00:00**；最晚截止：**2026-11-03T14:59:12+00:00**。
预测试：W=512 / 768 / 1024 各 5 轮均通过功能校验，按第 2–5 轮实测吞吐、版本落后及逐卡请求负载选择 **W=512**；这不等同于已证明训练质量更优。

最新 checkpoint：**100**，必需非空文件 **25/25**（仅文件存在性检查）。
最近 10 轮平均耗时：**134.33 秒/轮**（含验证；不含启动、停机空档）。

## 顺序停止与组大小

已闭合 **69866** 组：首条正确舍弃 **39407** 组（56.40%），8 条全错舍弃 **9443** 组（13.52%），有效完成 **21016** 组。
总舍弃率 **69.92%**；分母仅为实际已闭合组，不含 pending，不重复计入 carry/ready。
实际采样完成 **186122** 条；训练选中 **21015** 个完整组 / **71168** 条回答，组均 G=3.39。
口径：截至第 **139** 轮完整指标；不含未完成当前轮。未生成的回答长度未知，不把舍弃组数直接折算成 token 节省。

| 真实 G | 累计选中组数 |
|---:|---:|
| 2 | 9385 |
| 3 | 4397 |
| 4 | 2495 |
| 5 | 1728 |
| 6 | 1249 |
| 7 | 995 |
| 8 | 766 |

记录到的组/批次不变量：**passed**；无正式指标时显示 not_observed，而不是已通过。无需定向取消 canary：本方法不启动同题剩余请求，因此没有主动取消它们的步骤。

## 验证集

固定 DeepMath hard-dev **1,021 题**；AIME24 / AIME25 / AMC23 / MATH500 共 **600 题**。每题 4 次采样，报告 **mean@4**，不是 pass@4；unknown 计错。
最新完整测评：第 **100** 轮。

| 验证分组 | 题数 × 回答数 | mean@4 |
|---|---:|---:|
| DeepMath hard-dev | 1021 × 4 | 62.78% |
| DeepMath D5-6 | 128 × 4 | 67.77% |
| DeepMath D6-7 | 255 × 4 | 63.92% |
| DeepMath D7-8 | 319 × 4 | 63.17% |
| DeepMath D8-9 | 272 × 4 | 60.02% |
| DeepMath D9-10 | 47 × 4 | 56.38% |
| AIME24 | 30 × 4 | 20.00% |
| AIME25 | 30 × 4 | 12.50% |
| AMC23 | 40 × 4 | 64.38% |
| MATH500 | 500 × 4 | 83.55% |

![验证 mean@4](validation.svg?v=fb743eb06f6b)

hard-dev 按五个难度桶真实题数加权；四个公共测评不合成总分。仅连接实测检查点，无平滑。

<details>
<summary>各项历史最佳 mean@4</summary>

| 验证分组 | 最佳轮次 | mean@4 |
|---|---:|---:|
| DeepMath hard-dev | 100 | 62.78% |
| DeepMath D5-6 | 100 | 67.77% |
| DeepMath D6-7 | 100 | 63.92% |
| DeepMath D7-8 | 100 | 63.17% |
| DeepMath D8-9 | 100 | 60.02% |
| DeepMath D9-10 | 0 | 56.38% |
| AIME24 | 100 | 20.00% |
| AIME25 | 50 | 13.33% |
| AMC23 | 100 | 64.38% |
| MATH500 | 100 | 83.55% |

各项分别取最佳，不代表同一个 checkpoint 同时达到，也不保证该轮权重仍保留。

</details>

## 最近十轮

| 轮次 | 总秒数（含验证） | 采集秒数 | 更新秒数 | 真实组数 / 回答数 | 回答均长 | 跨权重回答 | stale tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 130 | 125.30 | 86.12 | 27.42 | 156 / 512 | 3604 | 41.21% | 65.32% |
| 131 | 137.70 | 92.82 | 31.01 | 163 / 512 | 4167 | 43.75% | 65.93% |
| 132 | 137.29 | 95.23 | 30.08 | 155 / 512 | 3912 | 39.06% | 66.12% |
| 133 | 126.74 | 83.61 | 30.09 | 150 / 512 | 3964 | 40.23% | 70.39% |
| 134 | 151.30 | 109.57 | 29.48 | 169 / 512 | 3890 | 35.16% | 52.88% |
| 135 | 128.34 | 86.77 | 28.84 | 162 / 512 | 3815 | 38.28% | 67.96% |
| 136 | 138.87 | 95.78 | 29.73 | 158 / 512 | 4043 | 36.91% | 61.93% |
| 137 | 129.42 | 85.63 | 30.66 | 160 / 512 | 4037 | 38.87% | 65.30% |
| 138 | 135.26 | 89.27 | 32.82 | 132 / 512 | 4228 | 42.58% | 70.41% |
| 139 | 133.06 | 85.89 | 33.65 | 166 / 512 | 4351 | 43.16% | 66.82% |

回答均长仅为选入训练的真实完整回答；跨权重比例指同一回答包含多个策略版本，stale token 比例指来自早于当前更新前策略的 token。collect 包括生成、续写、CPU 判分与队列处理，不是纯 decode。

## GPU 0–7

| GPU | 利用率 | 已用 / 总显存 GiB | 最新采样（UTC） | 状态 |
|---:|---:|---:|---|---|
| 0 | 100% | 189.57 / 268.59 | 2026-11-03T07:34:02+00:00 | ok |
| 1 | 100% | 190.50 / 268.59 | 2026-11-03T07:34:02+00:00 | ok |
| 2 | 100% | 190.50 / 268.59 | 2026-11-03T07:34:02+00:00 | ok |
| 3 | 100% | 190.50 / 268.59 | 2026-11-03T07:34:02+00:00 | ok |
| 4 | 100% | 190.50 / 268.59 | 2026-11-03T07:34:02+00:00 | ok |
| 5 | 100% | 190.50 / 268.59 | 2026-11-03T07:34:02+00:00 | ok |
| 6 | 100% | 190.56 / 268.59 | 2026-11-03T07:34:02+00:00 | ok |
| 7 | 100% | 190.50 / 268.59 | 2026-11-03T07:34:02+00:00 | ok |

## 最近十分钟 GPU 利用率

每秒采样，实际覆盖 **599.0/600 秒**；采样距页面生成 **0.8 秒**。缺测留空，无插值、无平滑。

![最近十分钟 GPU 利用率](gpu_utilization_10m.svg?v=b5a50a49f82d)

![最近十分钟 GPU 显存](gpu_memory_10m.svg?v=6e2ae7d10ae9)

[状态 JSON](status.json) · [验证 CSV](validation.csv) · [训练 CSV](training.csv) · [GPU 最新](gpu_latest.csv) · [GPU 秒级窗口](gpu_10m.csv) · [GPU 分钟历史](gpu_history.csv)

<details>
<summary>发布范围与历史口径</summary>

只发布汇总数字和图，不发布题目、回答、CoT、题目 ID、请求 ID、权重、设备标识或私有路径。校准集不参与训练；规则筛查不保证每道题金标无误。旧历史文件保持原样，当前独立曲线不并入任何旧组。

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
- [20261102-setup2-qwen3-8b-april-race4v2-w256-g64-8k-from0-remaining](history/20261102-setup2-qwen3-8b-april-race4v2-w256-g64-8k-from0-remaining/README.md)
- [20261103-setup2-qwen3-8b-april-seqfirst-w512-r512-8k-from0-10h-v2](history/20261103-setup2-qwen3-8b-april-seqfirst-w512-r512-8k-from0-10h-v2/README.md)
