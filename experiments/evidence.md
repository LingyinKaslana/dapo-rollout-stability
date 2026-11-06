# 逐项证据

每项链接到仓库内的聚合数据。来源指纹用于与保留在实验存储中的原始归档核对；不表示原始权重或逐题回答已随仓库发布。

<a id="c01"></a>

## C01 三次全参数更新的E/K/V增量谱

状态：已完成 局部证据。缓存变化小不代表增量可以高精度低秩拟合。

[聚合数据与来源指纹](../data/evidence/C01.json) · [研究章节](../docs/cache-reconstruction.md)

- 单seed、3次更新、4道重复heldout题；谱最多512抽样位置；oracle谱非在线算法。

<a id="c02"></a>

## C02 Wk/Wv实际更新的低秩近似

状态：已完成 局部证据。权重增量rank64留下约69–70%残差；图像似噪声不证明独立噪声。

[聚合数据与来源指纹](../data/evidence/C02.json) · [研究章节](../docs/cache-reconstruction.md)

- 这里是delta W，不是完整W；完整W的秩见C13/C15。

<a id="c03"></a>

## C03 Fourier与坐标函数拟合

状态：已完成 阴性结果。带符号增量的二维Fourier留出行误差接近100%；绝对值图规律不等于可预测符号。

[聚合数据与来源指纹](../data/evidence/C03.json) · [研究章节](../docs/cache-reconstruction.md)

- 同矩阵内插值，不是跨题泛化；非均匀实际token位置；adaptive FFT无heldout。

<a id="c04"></a>

## C04 仿射/ridge映射、末层修复、历史外推、量化稀疏

状态：已完成 局部证据。通道与逐头映射约改善28% KL；历史外推可劣于直接旧缓存。

[聚合数据与来源指纹](../data/evidence/C04.json) · [研究章节](../docs/cache-reconstruction.md)

- 4校准题和8评估题每对；main前缀2048；64固定后续token；FP32时延不能外推BF16生产性能。

<a id="c05"></a>

## C05 全网络一阶JVP与旧E新W

状态：已完成 局部证据。全JVP精度显著优于仅更新投影，但当前实现比新prefill慢约2.96倍。

[聚合数据与来源指纹](../data/evidence/C05.json) · [研究章节](../docs/cache-reconstruction.md)

- 24个前缀案例；参数增量为实际checkpoint差，不是裸梯度；oracle true E不可在线获得；能量汇总误差明显大于中位数。

<a id="c06"></a>

## C06 一阶与固定E误差的低秩残差

状态：Oracle 结构诊断。JVP残差较容易压缩，但依赖真实残差，尚未得到在线修正器。

[聚合数据与来源指纹](../data/evidence/C06.json) · [研究章节](../docs/cache-reconstruction.md)

- 6可视化案例×28层×512抽样位置；与C05全前缀误差不等口径。

<a id="c07"></a>

## C07 浅层E/V复用和K重算方案

状态：方案待在线验证。浅10层E/V变化能量占全层极少；尚未做完整混合方案在线长跑。

[聚合数据与来源指纹](../data/evidence/C07.json) · [研究章节](../docs/cache-reconstruction.md)

- early10_stale_only_oracle注入真实深层KV；不是计划中的旧E旧V新K算法。中高层入口状态须补充设计。

<a id="c08"></a>

## C08 0.6B GSM8K baseline/APRIL长跑权重轨迹

状态：部分完成。长跑已有训练曲线；本次已核对证据未包含多间隔E/K/V完整扫描。

[聚合数据与来源指纹](../data/evidence/C08.json) · [研究章节](../docs/cache-reconstruction.md)

- 原始配置包含机器路径，发布仅提取白名单字段；单次轨迹与反复验证不是多seed对比；APRIL不是我们的浅层E/V复用算法。

<a id="c09"></a>

## C09 相邻权重attention变化

状态：已完成 局部证据。平均attention TV很小但存在少量大变化query行。

[聚合数据与来源指纹](../data/evidence/C09.json) · [研究章节](../docs/cache-reconstruction.md)

- 只用4道heldout×3相邻更新；FP32 forward与FP64 attention，不是BF16 serving测量。

<a id="c10"></a>

## C10 旧top100重建及新top100 oracle

状态：已完成 阴性结果。旧/新top100误差几乎相同，固定100选择损失尾部质量主导误差。

[聚合数据与来源指纹](../data/evidence/C10.json) · [研究章节](../docs/cache-reconstruction.md)

- oracle新top100只有checkpoint001四题；后续32token固定teacher forcing；不支持kernel时延结论。

<a id="c11"></a>

## C11 按累计attention质量选择

状态：已完成 局部证据。99%/99.9%改善重建质量；99.9%仍需保留约46.48%的QK对。

[聚合数据与来源指纹](../data/evidence/C11.json) · [研究章节](../docs/cache-reconstruction.md)

- 当前为dense logits再mask；减少逻辑QK对不等于端到端省时；仅旧质量选择，新权重实际覆盖并非逐行保证。

<a id="c12"></a>

## C12 均值尾部补偿

状态：已完成 局部证据。top100+均值未稳定改善续写；99.9%+均值平均误差进一步下降。

[聚合数据与来源指纹](../data/evidence/C12.json) · [研究章节](../docs/cache-reconstruction.md)

- q/K/V来自新逐层近似状态；均值logit无法精确得到exp平均；尾部权重与V相关性被近似。

<a id="c13"></a>

## C13 Qwen GQA Wv秩

状态：已完成 局部证据。Wv为1024×2048且满行秩，一般E不能从V唯一恢复。

[聚合数据与来源指纹](../data/evidence/C13.json) · [研究章节](../docs/cache-reconstruction.md)

- 完整权重非增量；仅要求Q唯一时条件是ker(Wv)包含于ker(Wq)，方阵非必要条件。

<a id="c14"></a>

## C14 同权重旧KV恢复旧Q

状态：已完成 局部证据。FP32联合方向约束可近似恢复，BF16缓存量化使病态逆误差很大。

[聚合数据与来源指纹](../data/evidence/C14.json) · [研究章节](../docs/cache-reconstruction.md)

- 同checkpoint，不是跨权重；bf16_round是存储量化模拟，非原生BF16模型推理；raw K positive control使用额外未缓存信息；epsilon_inverse只在少量有效token有结果不能直接比较。

<a id="c15"></a>

## C15 Pythia无GQA的方阵Wv

状态：已完成 局部证据。FP64阈值满秩但高条件数；没有端到端恢复Q结果。

[聚合数据与来源指纹](../data/evidence/C15.json) · [研究章节](../docs/cache-reconstruction.md)

- Pythia-1.4B；FP32阈值数值秩2046–2047，不能把FP64满秩当稳定可逆。

<a id="c16"></a>

## C16 K/V相邻变化的口径复核

状态：已完成 局部证据。两模型被测相邻权重的K相对变化均小于V；绝对RMS和相对误差不同。

[聚合数据与来源指纹](../data/evidence/C16.json) · [研究章节](../docs/cache-reconstruction.md)

- Qwen1.7B本地1次RL更新间隔；Pythia公共1000次预训练更新间隔；两者不是可直接比较的训练对照；4固定文本前2048token。

<a id="c17"></a>

## C17 Q侧变化与局部attention敏感性

状态：已完成 局部证据。Q/K/V共同输入E已核对；分别替换旧Q/K/V改变attention输出；Q/K得分贡献范数不可相加。

[聚合数据与来源指纹](../data/evidence/C17.json) · [研究章节](../docs/cache-reconstruction.md)

- Q is an attention query, not a persistent autoregressive KV cache.
- Q/K/V projection inputs are the same tensor in each layer; pointer/shape equality is checked during every captured forward.
- Q statistics cover exactly the earlier 24 probe/update cases and 512 sampled positions, with full causal forwards of up to 8192 tokens.
- Q has 16 heads while K/V have 8 GQA heads. Concatenated Q dimension is 2048 and K/V dimension is 1024; per-head spectra compare equal 128-channel shapes.
- Q after RoPE is computed using the actual q_norm output and actual position embeddings from the same attention forward.
- Attention diagnostics sample 32 query positions per layer and retain all causal keys, mapping each KV head to its two query heads.
- Old/new Q/K/V are native components from old/new model forwards; component replacement isolates a local effect, not a complete hybrid-model forward or final answer.
- Q-term and K-term score ratios can exceed one because contributions can cancel; they are not percentages that sum to 100.
- Primary model compute is FP32 with TF32 disabled. Spectra and attention-score algebra use float64.
- Grayscale images show absolute deltas, shared/local 99.5th-percentile scales. Signed float32 delta arrays are retained.
- The four heldout questions recur across updates; observations are not independent questions. This study does not test training convergence.

<a id="g-01"></a>

## G-01 固定权重八同发与4+4生成微实验

状态：已完成 局部证据。固定权重4+4耗时为八同发的1.373倍；首四全对仍可能丢掉有效G8组。

[聚合数据与来源指纹](../data/evidence/G-01.json) · [研究章节](../docs/group-sampling.md)

- 无actor更新，无实际舍弃。
- 配对种子不保证批处理改变后的逐token相同轨迹。

<a id="g-02"></a>

## G-02 APRIL调度与pending窗口2×2短训

状态：已完成 局部证据。扩大pending窗口可补并发，但5轮筛选不证明稳态速度或质量收益。

[聚合数据与来源指纹](../data/evidence/G-02.json) · [研究章节](../docs/group-sampling.md)

- 未启用首四舍弃。
- 扩窗口同时改变并发和样本暴露；非稳态或质量证据。

<a id="g-03"></a>

## G-03 旧1.7B/4k八同发与4+4舍弃约20h长跑

状态：已完成 局部证据。旧1.7B设置的4+4多提交约6.4%轮次，不能外推到8B。

[聚合数据与来源指纹](../data/evidence/G-03.json) · [研究章节](../docs/group-sampling.md)

- 这里actor_updates字段是轮次提交计数，不是Adam优化器step。
- 不能向新8B数据与8k设置外推。

<a id="g-04"></a>

## G-04 8B APRIL eight、nodiscard44、discard44三组长跑

状态：已完成 局部证据。三组输出长度和工作量不同；质量没有单一排序。

[聚合数据与来源指纹](../data/evidence/G-04.json) · [研究章节](../docs/group-sampling.md)

- 均为APRIL跨轮续写，没有同步DAPO对照。
- 每组单轨迹，输出长度工作量不同。
- discard44轮503有actor提交但无完整轮次指标。

<a id="g-05"></a>

## G-05 8B四种组策略的长度演化

状态：已完成 局部证据。不同组策略都出现固定题评测长度增长，发生阶段不同。

[聚合数据与来源指纹](../data/evidence/G-05.json) · [研究章节](../docs/group-sampling.md)

- onset为预定义描述阈值，不是因果变点。
- race4v2轮335未完成记录不补造。

<a id="g-06"></a>

## G-06 race4 v1取消失效与排除

状态：实现无效 已排除。目标取消未生效；首版不得用于有效取消加速的性能结论。

[聚合数据与来源指纹](../data/evidence/G-06.json) · [研究章节](../docs/group-sampling.md)

- 底层请求ID映射与abort参数不匹配，筛选有效但取消无效。
- 不得作为成功取消加速器的结果发布。

<a id="g-07"></a>

## G-07 seqfirst串行至首个正确的可变组实验

状态：核验或复核。串行至首次正确保留可变大小0…01组，改变了组分布。

[聚合数据与来源指纹](../data/evidence/G-07.json) · [研究章节](../docs/group-sampling.md)

- 只保留0...01；首答即对和8次全错舍弃。
- selection_correction=none；不是full8对照。
- 本索引不推断正式轨迹最终停止轮数。

<a id="g-08"></a>

## G-08 0.6B GSM8K三策略1000轮稳定性筛选

状态：已完成 局部证据。三组各1000轮均未触发既定失稳筛选门槛，单种子。

[聚合数据与来源指纹](../data/evidence/G-08.json) · [研究章节](../docs/group-sampling.md)

- 模型、数据、上限、window和batch不同于8B。
- 不稳定阈值是探索筛选规则，不是显著性检验。
- 初始设计JSON状态过期，以11/3完成快照为准。

<a id="g-09"></a>

## G-09 超过已正确短回答2倍长度时停止的离线估算

状态：离线估算。离线2L停止能省token，也会截掉原本正确的回答。

[聚合数据与来源指纹](../data/evidence/G-09.json) · [研究章节](../docs/group-sampling.md)

- Reference is the shortest already correctly scored response.
- Equal token-generation rates within each concurrently started stage; reward latency is zero.
- Current 4+4 starts stage two only after the first four settle; later correct references never retroactively truncate stage one.
- Cancel after floor(2*reference_length)+1 generated tokens only if the response has not finished.
- Fixed observed generation traces: weight updates, sampling, future training and wall-clock speed effects are not simulated.
- Originally correct aborted means the full original answer was correct; the shortened answer is not regraded.

<a id="g-10"></a>

## G-10 APRIL跨轮跨度与重复前缀重建统计

状态：已完成 局部证据。反复短续写再暂停不是该快照主流；仍有前缀重建开销。

[聚合数据与来源指纹](../data/evidence/G-10.json) · [研究章节](../docs/group-sampling.md)

- 来自discard44的一份快照，不代表全部实验。
- Native call timing does not separate queue/prefill/decode; no wall-time fraction attributable to rebuild is identifiable.

<a id="g-11"></a>

## G-11 8B正式setup迁移与生成/反向基础检查

状态：已完成 局部证据。设置与奖励链经过基础核验；基准就绪不意味着算法更优。

[聚合数据与来源指纹](../data/evidence/G-11.json) · [研究章节](../docs/group-sampling.md)

- ARM64单卡重新搭建环境，未直接复制x86环境。
- 迁移检查仅生成执行与CE反向检查，不是完整RL收敛验证。

<a id="g-12"></a>

## G-12 Base与discard44 step503的首中次数和采样覆盖

状态：已完成 局部证据。首答和100次覆盖提高，总生成token却增加约85.2%。

[聚合数据与来源指纹](../data/evidence/G-12.json) · [研究章节](../docs/group-sampling.md)

- 目录名含20，但最终上限实际为100。
- 旧1621题评测集非新盲测。
- 一次配对种子方案，有限覆盖不足以证明新能力或能力支持集不变。
- 不得公开逐题CSV：含题目标识，改用聚合首中曲线。

<a id="g-13"></a>

## G-13 8B候选与组内rollout长度分布

状态：已完成 局部证据。回答长度与组内最大值呈长尾，训练选中组与全部候选分母不同。

[聚合数据与来源指纹](../data/evidence/G-13.json) · [研究章节](../docs/group-sampling.md)

- 仅完整候选组；未完成的组被截尾。
- token-only同步利用率只是代理量，不是实测GPU利用率。

<a id="s01"></a>

## S01 实验设置与指标边界

状态：核验或复核。early formal、late nodiscard44、full8 eight分别处理；裁剪前范数不同于Adam参数更新幅度。

[聚合数据与来源指纹](../data/evidence/S01.json) · [研究章节](../docs/training-stability.md)

- online branches sampled different data; fixed replay intentionally bypasses async collection
- 研究中的baseline可能指原未校正APRIL，而不是同步DAPO；original eight在663的运行错误不能已归因于大梯度。

<a id="s02"></a>

## S02 早期首token直接损失干预

状态：受控干预。首token置零降低端点自然长度与Alright比例，随机/NLLmatched单token对照无相同效果。

[聚合数据与来源指纹](../data/evidence/S02.json) · [研究章节](../docs/training-stability.md)

- 固定历史batch训练后评测，不是在线采集训练；端点同配置seed非per-request paired RNG；正文共享梯度仍可更新首token偏好。

<a id="s03"></a>

## S03 Base与早期checkpoint的强制开头正确率和方差

状态：已完成 局部证据。Base的Alright正确率高于To；G4方差差异的区间跨零，不能推出更好梯度信号。

[聚合数据与来源指纹](../data/evidence/S03.json) · [研究章节](../docs/training-stability.md)

- These are forced-prefix evaluations on one64-question panel, T=1, max8192; not actual G8 training group statistics.
- Conditional Bernoulli variance p(1-p) is tied to conditional accuracy. Across-question heterogeneity is a different quantity.
- Actual shaped reward includes an overlong penalty. Binary variance does not measure full reward variance or parameter-gradient variance.
- More nonzero advantages do not imply positive reinforcement of a style. Natural historical opening accuracy remains selection-confounded; n=0 is missing, never zero accuracy.
- No same-panel forced-opening later checkpoint comparison is available in the existing evaluated results.

<a id="s04"></a>

## S04 实际训练中开头收益和长度惩罚的信号反转

状态：已完成 局部证据。early Alright平均优势为正，late另一轨迹为负；移除长度惩罚仍负，正确率/内容也在改变。

[聚合数据与来源指纹](../data/evidence/S04.json) · [研究章节](../docs/training-stability.md)

- Early formal and late500 continuation are different trajectories. Do not splice them into a cycle.
- Mixed-group comparisons condition on the same question and selected group, but opening choice, generation histories, and content are not randomized.
- A/D and even the exact scalar dLoss/dlogp coefficients are not the final change of opening probability: softmax coupling, shared body gradients, clipping, and Adam remain.
- Native actual first-token coefficients cover only8 selected updates. Their sums cannot stand in for all training updates.
- Later very rare Alright groups have small n. Group centering makes the dominant style's mean A approach zero without proving its relative reward advantage vanished.

<a id="s05"></a>

## S05 Base一万条异常筛选

状态：已完成 局部证据。Base已有乱码、漂移、对话与重复；单次输出异常不等于题目稳定诱发。

[聚合数据与来源指纹](../data/evidence/S05.json) · [研究章节](../docs/training-stability.md)

- AI review is not a human gold standard.
- A single response is observed per prompt; this does not estimate per-prompt repeatability.
- No correctness or nonzero-advantage filter was applied.
- Partial reviewed subsets may be selection biased; counts remain snapshot counts.
- Types overlap; capped status alone is never an anomaly verdict.

<a id="s06"></a>

## S06 Base长度与重复、盲审误报复核

状态：核验或复核。更长回答有更高重复标签率，但因果方向未识别；盲审发现正常进展与轻度冗余被混入。

[聚合数据与来源指纹](../data/evidence/S06.json) · [研究章节](../docs/training-stability.md)

- Generated length is post-outcome; association does not establish long output causes loops.
- Different questions in each bin, one draw per question; cannot estimate per-question persistent risk.
- LLM labeling can falsely flag legitimate checking/alternate reasoning; examples and independent review required.
- Natural opening comparisons are confounded by problem selection and generation length.
- 八条盲审样例经过选择，不能当成代表总体的误报率；不发布原题/回答/案例文本。

<a id="s07"></a>

## S07 异常前缀、checkpoint与数据触发检查

状态：受控干预。相同失败前缀更换checkpoint强烈改变循环；局部重采样支持题目×权重×轨迹交互而非单一坏题解释。

[聚合数据与来源指纹](../data/evidence/S07.json) · [研究章节](../docs/training-stability.md)

- Four failure-selected trajectories from only three questions, including two trajectories from the same math problem; not an estimate of population failure rate.
- Rewind and fresh-prompt arms begin earlier and use the same next2048-token horizon, so they do not establish full-response eventual failure rates.
- The non-@ alternative was selected by FP32 token probability excluding special tokens; in some cases it is extremely unlikely and not guaranteed to preserve mathematical semantics.
- Keep500 has0/32 @ repetition at the boundary, but21/32 continuations still hit the2048-token cap; correctness has not been evaluated.
- Attribution targets three previously selected high-mismatch tokens and uses FP32. Input-gradient concentration on shared prompt/template tokens is descriptive, not proof that the template causes instability.
- This addresses conditional generation and local sensitivity at late checkpoints. No training update ablation or causal reproduction of the earlier length surge was performed; those early full checkpoints were not retained.
- Six evaluation prompts, selected using earlier diagnostic extremes and low-mismatch controls; not a random training-data sample.
- Cross-checkpoint fixed-token comparisons use vLLM prefill vs HF full sequence; new sampled responses use recorded native decode vs HF.
- BF16/FP16 use the same exported BF16 weight values. FP32 reference additionally changes attention from FA2 to SDPA and uses exact prefixes; verified BF16/FP16 prefix and full-sequence scores agree at the selected points.
- Gradient probes use unit-token negative log probability with no advantages, token averaging, PPO clipping, optimizer state, or updates.
- Checkpoints 500 and 541 concern late instability, not race4v2 steps126-140. No training-level causal ablation was performed.

<a id="s08"></a>

## S08 屏蔽@的缓解尝试

状态：已完成 阴性结果。符号屏蔽消除@但没有恢复失败边界的正常停止或准确率，可能换为其他重复。

[聚合数据与来源指纹](../data/evidence/S08.json) · [研究章节](../docs/training-stability.md)

- Late checkpoints 500/541, not the early length surge.
- Four selected failure trajectories do not estimate population prevalence.
- 64 dev questions have limited power for rare failures.
- Hard masking may remove almost all probability at a boundary; rescue would not identify the training cause.

<a id="s09"></a>

## S09 PPL尾部与时间顺序

状态：核验或复核。算术平均PPL可被极少回答主导；不能从总曲线推断单token触发或普通回答整体恶化。

[聚合数据与来源指纹](../data/evidence/S09.json) · [研究章节](../docs/training-stability.md)

- These metrics score selected sampled answers, not a fixed heldout corpus; content/policy versions and length vary.
- Arithmetic PPL can be dominated by a few responses; geometric PPL and median are different diagnostics, not replacements for the logged metric.
- Opening-conditioned PPL rows are observational and can have very small n. PPL is not correctness variance or parameter-gradient variance.

<a id="s10"></a>

## S10 负优势校正和token权重对应关系

状态：受控干预。负优势TIS在共同10轮抑制峰值；20轮中正确token对应的TIS显著优于均值/乱序权重。

[聚合数据与来源指纹](../data/evidence/S10.json) · [研究章节](../docs/training-stability.md)

- 单个历史流与一个乱序seed；TIS cap2不是无偏整轨迹校正；均值/打乱分支无行为端点评测；PPO后系数不保证总质量守恒。

<a id="s11"></a>

## S11 最高NLL/失配token的梯度净投影

状态：已完成 局部证据。最高少量token并未解释主要净梯度，低p0/q的大集合贡献约59%。

[聚合数据与来源指纹](../data/evidence/S11.json) · [研究章节](../docs/training-stability.md)

- 517为重构轮内点非原生产峰值；这是集合净投影，允许内部抵消；不排除前缀内容或更早触发。

<a id="s12"></a>

## S12 固定信号下的状态响应、FP32和块回退

状态：受控干预。同一输入与固定系数的梯度随模型状态增大；FP32 master仍大；部分中间层回退降低响应但未验证修复质量。

[聚合数据与来源指纹](../data/evidence/S12.json) · [研究章节](../docs/training-stability.md)

- 固定事后挑选的8回答与固定系数，非全局Jacobian或前瞻预警；BF16参数像转FP32与真实master要分开；单点FP32不代表FP32全训练。

<a id="s13"></a>

## S13 full8 eight长度、开头与后期梯度

状态：已完成 局部证据。350→400长度翻倍而梯度仍小；600→650评测长度和机械重复下降但梯度上涨。

[聚合数据与来源指纹](../data/evidence/S13.json) · [研究章节](../docs/training-stability.md)

- Only this eight trajectory is plotted; no splice with formal or nodiscard44.
- Natural opening strata are selected by the policy. Sparse pre-transition Alright cannot identify a causal composition effect.
- Validation sampling T0.6/top_p0.95/top_k20 differs from training T1/p1/k-1; not the same prompts/responses as training batches.
- Mechanical rule is conservative, not semantic-loop or gibberish prevalence. Rule-negative does not certify normal output.
- PPL here uses every10-round native metric samples. Opening/validation points are every50 rounds. Within-interval lead/lag is not resolved.
- Original350/400 actor checkpoints were pruned; only600/650 remain complete per the latest inventory.

<a id="s14"></a>

## S14 在线TIS/同步分支与输出质量

状态：在线单轨迹对照。从500启用TIS大幅压低梯度，但64题端点异常标签更多；低梯度不保证好质量。

[聚合数据与来源指纹](../data/evidence/S14.json) · [研究章节](../docs/training-stability.md)

- 不同在线分支回答不同，非同样本因果对照；64题T1不同于1621题T0.6固定验证；AI标签非人工真值；原500无driver Python/NumPy RNG，非位级历史续跑。

<a id="s15"></a>

后续更新：650→850已完成；以下是原启动阶段快照。[完整后续报告](../reports/rl-length-gradient-investigation.md)。

## S15 full8 650→850续训

状态：运行中。从完整650恢复，继续200轮；首轮651验证通过，后期趋势待运行结束后分析。

[聚合数据与来源指纹](../data/evidence/S15.json) · [研究章节](../docs/training-stability.md)

- 状态为本地采集快照，不宣称实时进度或已完成；不能把重跑651–662当新增历史。

<!-- final-stability-progress:start -->
## S16

[同一600状态的八组稳定性干预](final-stability-600-650.md)当前纳入4/8组训练、5/9个次要评测端点。已纳入四组的入训与候选统计均为全50轮，整个八组实验仍未完成，见[聚合证据](../data/evidence/S16.json)。
<!-- final-stability-progress:end -->
