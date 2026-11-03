# Qwen3-8B Base APRIL sequential-first-success + DAPO 训练进度

更新时间：2026-11-03T11:44:02+00:00（UTC，计划每 10 分钟同步；网络失败时顺延）

方法：同一题每次只生成 **1 条独立回答**，判分后若错误则继续；首次成功或最多 8 次后停止。首条就正确的 G1 组与 8 条全错组不训练；有效组为 **×…×√，G=2–8**。后一次仍输入原题，不把前次错误回答拼进 prompt。

**8 张 GPU · pending 窗口 W=512 组 · 每次更新恰好 512 条真实回答 · 有效组数可变**。同一题固定路由到同一张 GPU，沿用同权重的原生前缀缓存；不复用跨权重 KV。并行的是不同题，每题最多一条回答在途。

按完整有效组装箱；不截组、不重复回答、不补假奖励。ready 池独立，剩余完整组留到后续。原生 optimizer 仍为 **4 × 128 条回答**；配置中的 nominal n=8 / train_batch=64 / mini_batch=16 只是训练几何，不代表每轮选中 64 组或每题实际生成 8 条。

独立实验：从同一个 Qwen3-8B-Base 重新开始，全参数训练。总上限 **10 小时（80 GPUh）**，包含初始化、训练、测评、保存与收尾；**9 小时 45 分开始边界收尾**，不追加预算。

DeepMath open-v4 train 40,299 · 回答上限 8192 tokens · 学习率 5e-7。启动时初始评估，之后每 **50 轮评估**、每 **50 轮保存**，保留最近 2 个检查点。

**训练中 · 已记录 299 轮完整训练指标 · 实际分配 6.75 小时**

当前轮次：300；阶段：验证。
收尾时间：**2026-11-03T14:44:12+00:00**；最晚截止：**2026-11-03T14:59:12+00:00**。
预测试：W=512 / 768 / 1024 各 5 轮均通过功能校验，按第 2–5 轮实测吞吐、版本落后及逐卡请求负载选择 **W=512**；这不等同于已证明训练质量更优。

最新 checkpoint：**300**，必需非空文件 **25/25**（仅文件存在性检查）。
最近 10 轮平均耗时：**32.17 秒/轮**（含验证；不含启动、停机空档）。

## 顺序停止与组大小

已闭合 **167961** 组：首条正确舍弃 **100471** 组（59.82%），8 条全错舍弃 **21754** 组（12.95%），有效完成 **45736** 组。
总舍弃率 **72.77%**；分母仅为实际已闭合组，不含 pending，不重复计入 carry/ready。
实际采样完成 **427596** 条；训练选中 **45735** 个完整组 / **153088** 条回答，组均 G=3.35。
口径：截至第 **299** 轮完整指标；不含未完成当前轮。未生成的回答长度未知，不把舍弃组数直接折算成 token 节省。

| 真实 G | 累计选中组数 |
|---:|---:|
| 2 | 21058 |
| 3 | 9293 |
| 4 | 5369 |
| 5 | 3726 |
| 6 | 2628 |
| 7 | 2069 |
| 8 | 1592 |

记录到的组/批次不变量：**passed**；无正式指标时显示 not_observed，而不是已通过。无需定向取消 canary：本方法不启动同题剩余请求，因此没有主动取消它们的步骤。

## 验证集

固定 DeepMath hard-dev **1,021 题**；AIME24 / AIME25 / AMC23 / MATH500 共 **600 题**。每题 4 次采样，报告 **mean@4**，不是 pass@4；unknown 计错。
最新完整测评：第 **250** 轮。

| 验证分组 | 题数 × 回答数 | mean@4 |
|---|---:|---:|
| DeepMath hard-dev | 1021 × 4 | 65.57% |
| DeepMath D5-6 | 128 × 4 | 66.41% |
| DeepMath D6-7 | 255 × 4 | 65.78% |
| DeepMath D7-8 | 319 × 4 | 65.91% |
| DeepMath D8-9 | 272 × 4 | 65.26% |
| DeepMath D9-10 | 47 × 4 | 61.70% |
| AIME24 | 30 × 4 | 18.33% |
| AIME25 | 30 × 4 | 20.00% |
| AMC23 | 40 × 4 | 62.50% |
| MATH500 | 500 × 4 | 83.50% |

![验证 mean@4](validation.svg?v=f11302a0a95f)

hard-dev 按五个难度桶真实题数加权；四个公共测评不合成总分。仅连接实测检查点，无平滑。

<details>
<summary>各项历史最佳 mean@4</summary>

| 验证分组 | 最佳轮次 | mean@4 |
|---|---:|---:|
| DeepMath hard-dev | 200 | 70.35% |
| DeepMath D5-6 | 150 | 74.22% |
| DeepMath D6-7 | 150 | 72.16% |
| DeepMath D7-8 | 200 | 72.96% |
| DeepMath D8-9 | 150 | 66.82% |
| DeepMath D9-10 | 150 | 62.23% |
| AIME24 | 200 | 25.00% |
| AIME25 | 250 | 20.00% |
| AMC23 | 200 | 65.00% |
| MATH500 | 200 | 86.30% |

各项分别取最佳，不代表同一个 checkpoint 同时达到，也不保证该轮权重仍保留。

</details>

## 最近十轮

| 轮次 | 总秒数（含验证） | 采集秒数 | 更新秒数 | 真实组数 / 回答数 | 回答均长 | 跨权重回答 | stale tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 290 | 28.90 | 15.57 | 6.68 | 146 / 512 | 598 | 24.02% | 58.39% |
| 291 | 30.05 | 16.33 | 7.20 | 144 / 512 | 584 | 23.05% | 61.12% |
| 292 | 30.12 | 17.19 | 6.62 | 125 / 512 | 592 | 23.05% | 69.63% |
| 293 | 27.25 | 13.84 | 6.95 | 137 / 512 | 594 | 22.27% | 70.14% |
| 294 | 28.09 | 14.97 | 6.16 | 138 / 512 | 597 | 24.61% | 61.23% |
| 295 | 29.72 | 16.12 | 6.67 | 145 / 512 | 680 | 26.56% | 69.14% |
| 296 | 30.51 | 16.49 | 7.09 | 151 / 512 | 628 | 21.48% | 61.81% |
| 297 | 29.44 | 16.14 | 7.04 | 150 / 512 | 663 | 23.05% | 61.44% |
| 298 | 36.90 | 21.66 | 8.51 | 152 / 512 | 749 | 24.80% | 53.26% |
| 299 | 50.71 | 35.19 | 8.05 | 151 / 512 | 967 | 25.20% | 50.06% |

回答均长仅为选入训练的真实完整回答；跨权重比例指同一回答包含多个策略版本，stale token 比例指来自早于当前更新前策略的 token。collect 包括生成、续写、CPU 判分与队列处理，不是纯 decode。

## GPU 0–7

| GPU | 利用率 | 已用 / 总显存 GiB | 最新采样（UTC） | 状态 |
|---:|---:|---:|---|---|
| 0 | 99% | 189.03 / 268.59 | 2026-11-03T11:44:01+00:00 | ok |
| 1 | 94% | 190.09 / 268.59 | 2026-11-03T11:44:01+00:00 | ok |
| 2 | 97% | 190.06 / 268.59 | 2026-11-03T11:44:01+00:00 | ok |
| 3 | 97% | 190.12 / 268.59 | 2026-11-03T11:44:01+00:00 | ok |
| 4 | 99% | 190.08 / 268.59 | 2026-11-03T11:44:01+00:00 | ok |
| 5 | 100% | 190.12 / 268.59 | 2026-11-03T11:44:01+00:00 | ok |
| 6 | 98% | 189.98 / 268.59 | 2026-11-03T11:44:01+00:00 | ok |
| 7 | 99% | 190.25 / 268.59 | 2026-11-03T11:44:01+00:00 | ok |

## 最近十分钟 GPU 利用率

每秒采样，实际覆盖 **599.0/600 秒**；采样距页面生成 **0.7 秒**。缺测留空，无插值、无平滑。

![最近十分钟 GPU 利用率](gpu_utilization_10m.svg?v=a6fedc9670db)

![最近十分钟 GPU 显存](gpu_memory_10m.svg?v=5bcd9c13db21)

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
