# 梯度稳定性实验：损失权重与重复定位参考实现

这里整理600→650八组对照实际使用的六个核心模块。六个文件与成功运行时的冻结文件逐字节一致，来源和发布字节的SHA256见[source_manifest.json](source_manifest.json)。本目录提供CPU可检验的算法接口；实验结果应从研究报告读取，本目录不包含结果、权重或原始问答。

## 八组处理的精确定义

记原回答的有效位置为 $m_{i,t}$，原优势为 $A_{i,t}$。本实验的GRPO优势在每条回答内恒定，但“负”始终指 $A<0$，不能直接替换成“答错”。令

$$
s_{i,t}=\log p_{0,i,t}-\log q_{i,t},\qquad
D_B=\sum_{i\in B,t}m_{i,t}.
$$

$p_0$ 是这一轮更新前、训练网络对入训回答算出的概率；$q$ 是生成时保存的历史行为概率。四次原生更新共用冻结的 $p_0$ 和干预权重。每次更新的当前训练概率 $p_\theta$ 会变化，诊断中的 $\log p_\theta-\log q$ 与上述固定门控量分别统计。

所有组保留原来的PPO裁剪、dual-clip、优势、上下文、response mask和物理分母。干预只在原生逐token损失聚合前乘上脱离计算图的 $w_{i,t}$：

$$
L_B=\frac{1}{D_B}\sum_{i\in B,t}m_{i,t}w_{i,t}
\ell_{\rm native}(\log p_\theta,\log p_0,A).
$$

原生分布式执行器继续负责DP缩放；分母不替换成 $\sum mw$，也不改成保留token数。

| 组名 | $w_{i,t}$ |
|---|---|
| `baseline` | 1 |
| `full_tis` | $\min[2,\exp(\operatorname{clip}(s^{32},-20,20))]$ |
| `negative_tis` | $A<0$ 时使用上述TIS权重，其余为1 |
| `abs_gap_mask` | $1-\mathbf1[\lvert s^{64}\rvert>0.5]$ |
| `negative_down_mask` | $1-\mathbf1[A<0\land s^{64}<-0.5]$ |
| `repeat_negative_down_mask` | $1-\mathbf1[A<0\land s^{64}<-0.5\land R_{i,t}]$ |
| `negative_mass` | 负优势回答取 $\lambda_B\min(1,2048/T_i)$，其余为1 |
| `sync_fresh` | 权重为1；采集器另行改为使用当前权重完整生成新的G8回答 |

TIS使用FP32操作数相减。这里的TIS下限为 $\exp(-20)\approx2.061\times10^{-9}$，这是相对早期仅设上限的负优势TIS实现的变化，需要记录floor命中数。某次数据中未命中，不能证明这个差异在所有情况下都无关。原生PPO自身对当前/旧策略log-ratio的数值保护是另一个独立步骤。

三个硬门控先分别转FP64，再做减法和比较。这样，两个已存FP32数的真实差值若刚好超过0.5，不会因FP32减法舍入回0.5。门槛严格使用 `>` / `<`，恰好等于正负0.5时保留。输出权重为detached FP32；padding位置强制为1，由原mask排除。

方法6按**每个原始全局128条子批次**分别匹配负优势质量：

$$
u_i=\min(1,2048/T_i),\quad
\lambda_B=\frac{\sum_{i\in B,A_i<0}|A_i|T_i}
{\sum_{i\in B,A_i<0}|A_i|T_i u_i}.
$$

没有负优势时取1。它保持的是原始anchor的 $\sum|A|T$，不是经过PPO裁剪后的梯度质量。较短的负优势回答可得到大于1的权重。原生512条回答先完成负载平衡，然后每rank64条、四次各16条；默认分组函数据此形成四个全局128条子批次。四组互不重叠；不是把同一批512条反复训练四次。其他布局需要显式传入真实的全局分组，不能在各rank或microbatch内单独归一化。

方法7需要外部采集器实现。本模块只返回其损失乘数1；它不会把一批旧回答变成新数据。同步生成也不能保证训练与推理的数值概率完全相同，B2–B4的参数更新仍存在。

## 重复位置 $R_{i,t}$

本实验使用 `exact confirmed mask OR strict noisy mask`，由[batch_text_metrics.py](batch_text_metrics.py)组合。两部分都只利用原token IDs及原题目文本，不读取概率、优势、奖励或最终正确性。

- **精确循环部分**：逐字符NFKC、去空白，保留大小写；周期1–16。字母循环要求此前至少128个归一化字符及16个周期；数字/符号循环还要求此前至少512字符、256个可见token及16个周期。只标记继续已有循环的当前位置，覆盖所有满足条件的片段，没有早期辅助loss的64位置上限。
- **严格局部复用部分**：当前token之前512个归一化字符分成两个256字符窗口。每段含253个四字符片段，要求两段的 $1-\#\text{不同四字符片段}/253\ge0.75$。当前token新形成的四字符片段至少一半，要在旧512字符窗口中各出现至少两次。它不额外要求256个可见token。

[noisy_repetition_mask.py](noisy_repetition_mask.py)仍保留旧命名：`primary_mask`使用0.5、`strict_mask`使用0.75；本实验方法5启用后者。不要把函数内部的`primary_mask`误认为本实验启用的门槛。

EOS/特殊token、空白、无法可靠对齐的边界及replacement character等受到排除。题目明确要求重复、某些长整数/小数/矩阵推导有保护；“只求末几位”等情况有更细的规则。这些都是启发式保护，不能证明所有命中均异常或所有异常均被覆盖。

用于监测的128字符全类型循环候选，与用于损失的因果确认mask分开。例如200个数字字符可被监测为循环，却未达到保守精确mask的训练门槛。原文本中的token位置始终保留，不对可见文本重新分词后替换位置。`batch_text_metrics`核验冻结的单user ChatML提示模板；其他模板需要单独适配并记录新版本。较长周期的语句循环仍可能漏检；`strict_rules`的监测标记不会自动扩大方法5的损失mask。

## 接入原生PPO

本实验实际 `native_config` 使用 `clip_ratio_low=0.2`、`clip_ratio_high=0.28`、`clip_ratio_c=10.0`。原生PPO比值为 $r=\exp(\operatorname{clip}(\log p_\theta-\log p_0,-20,20))$：分母 $p_0$ 是当前轮更新前由训练actor重新计算、在四次更新内保持的概率；历史生成概率 $q$ 另行保留，用于干预权重和诊断。baseline的原生IS乘数槽为 `None`，不能把baseline比值写成 $p_\theta/q$。CPU测试中的 `clip_ratio_c=3.0` 仅用于合成分支覆盖，**实际实验的dual-clip参数是10.0**。

一次outer round开始时，在**最终平衡后的原始行序**上准备权重，并通过原生`rollout_is_weights`参数传入：

```python
from loss_variants import make_loss_weights, negative_mass_weights

supplied = None
if arm == "negative_mass":
    supplied, audit = negative_mass_weights(A, response_mask, groups=global_groups)

weights = make_loss_weights(
    arm, old_log_probs, rollout_log_probs, A, response_mask,
    repeat_mask=original_response_repeat_mask,
    supplied_weights=supplied,
)

# 分布式packing后，取与当前microbatch同一行/同一位置的weights。
loss, metrics = native_policy_loss(
    old_log_prob=old_log_probs,
    log_prob=current_log_probs,
    advantages=A,
    response_mask=response_mask,
    loss_agg_mode="token-mean",
    config=native_config,
    rollout_is_weights=None if arm in ("baseline", "sync_fresh") else weights,
)
loss.backward()
```

这段接口示例省略了分布式切分。baseline和sync_fresh保持原来的空乘数槽；其他组接入时也要确认该乘数槽原先为空，避免与另一层rollout IS重复相乘。实际接入需要核验：原A和mask未改；p0在轮内冻结；权重与原response坐标严格对齐；物理D不变；只执行原来的反向传播；零权重位置的直接 `dL/dlogp` 为0。被屏蔽的位置仍可通过后续token的上下文间接产生参数梯度。

[native_ppo_reference.py](native_ppo_reference.py)是本次冻结原生PPO两个函数的CPU抽取，保留原函数体和版权声明，帮助检验权重是否作用在正确位置。它使用[native_masked_reference.py](native_masked_reference.py)中逐函数原样抽取的原生masked sum/mean，包括先排除mask外NaN和均值分母的1e-8项。全padding的本地分片在有效的全局token分母下前向损失为0；若全局分母本身为0，聚合器仍保留原生的NaN行为。这里没有把前向测试扩展宣称为NaN输入的反向安全保证。该参考代码不负责分布式训练、恢复、rollout或GPU内核选择。

## 指标与计数

`loss_variants.py`提供可跨microbatch/rank相加的充分统计；先求和，再计算均值与RMS。$d=|\log p-\log q|$不是PPL或全词表KL。报告超0.5的占比与条件均值；没有尾部token时条件均值为0，缺失的人群指标仍保留缺失。

- L2为 $\sqrt{\sum d_t^2}$，受token数影响；RMS为 $\sqrt{\sum d_t^2/N}$。
- log-PPL为token加权平均NLL，PPL是其指数，不是逐token PPL的算术平均。
- 原token位置数、权重和、$\sum|A|w$、实际直接损失系数绝对质量，以及参数梯度范数分别报告。改变权重不等于删除一次回答使用，也不改变物理D。
- 实际Adam位移与裁剪前梯度范数分别测量；后者降低不能自动推导前者或解题质量改善。
- 入训回答、新完成rollout、独立验证三个人群分开。只保存q的all-completed回答，没有实际actor forward就没有其d。
- `first_word_alright/to`是可见首个ASCII词的频率，不是模型首token概率；原首token ID/piece另有字段。
- 生成耗时、训练耗时、token条次与加权质量不是同一种量；CPU文本评估若已包含在actor耗时内，不再次加到总耗时。

## CPU验证

需要Python、NumPy、PyTorch。使用真实本地tokenizer的持久进程池还需要Transformers；下列合成契约测试使用确定性的字符tokenizer，不下载模型、不调用API、不启动GPU：

```bash
python run_cpu_tests.py
```

测试覆盖正负边界与FP64硬门控、TIS下限/上限和detach、全局子批次质量匹配、原生裁剪/dual-clip分支上的直接梯度、因果前缀一致性、EOS与decoder别名、正常数学保护、原始padding坐标、全padding/含NaN的padding前向聚合、spawn worker顺序和指标分母。合成测试数据不作为研究结果。原始GPU恢复、FA2实现和真实packing仍需要完整训练环境的独立验收。

原生PPO抽取遵循[Apache-2.0许可证](LICENSE.native-ppo)，来源标识与完整源文件SHA记录在[source_manifest.json](source_manifest.json)。
