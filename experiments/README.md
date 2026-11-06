# 实验与证据索引

[长度/梯度/循环后续52项证据](stability-investigation.md)。

这些是研究单元，不是独立训练重复的计数。同一个单元可以包含多个对照；进行中、无效和方案状态单独标注。详见[设置对照](../docs/setup-comparison.md)。

## 缓存重建

| 编号 | 实验与问题 | 状态 | 结果与限定 |
|---|---|---|---|
| [C01](evidence.md#c01) | [三次全参数更新的E/K/V增量谱](../docs/cache-reconstruction.md) | 已完成 局部证据 | 缓存变化小不代表增量可以高精度低秩拟合。 |
| [C02](evidence.md#c02) | [Wk/Wv实际更新的低秩近似](../docs/cache-reconstruction.md) | 已完成 局部证据 | 权重增量rank64留下约69–70%残差；图像似噪声不证明独立噪声。 |
| [C03](evidence.md#c03) | [Fourier与坐标函数拟合](../docs/cache-reconstruction.md) | 已完成 阴性结果 | 带符号增量的二维Fourier留出行误差接近100%；绝对值图规律不等于可预测符号。 |
| [C04](evidence.md#c04) | [仿射/ridge映射、末层修复、历史外推、量化稀疏](../docs/cache-reconstruction.md) | 已完成 局部证据 | 通道与逐头映射约改善28% KL；历史外推可劣于直接旧缓存。 |
| [C05](evidence.md#c05) | [全网络一阶JVP与旧E新W](../docs/cache-reconstruction.md) | 已完成 局部证据 | 全JVP精度显著优于仅更新投影，但当前实现比新prefill慢约2.96倍。 |
| [C06](evidence.md#c06) | [一阶与固定E误差的低秩残差](../docs/cache-reconstruction.md) | Oracle 结构诊断 | JVP残差较容易压缩，但依赖真实残差，尚未得到在线修正器。 |
| [C07](evidence.md#c07) | [浅层E/V复用和K重算方案](../docs/cache-reconstruction.md) | 方案待在线验证 | 浅10层E/V变化能量占全层极少；尚未做完整混合方案在线长跑。 |
| [C08](evidence.md#c08) | [0.6B GSM8K baseline/APRIL长跑权重轨迹](../docs/cache-reconstruction.md) | 部分完成 | 长跑已有训练曲线；本次已核对证据未包含多间隔E/K/V完整扫描。 |
| [C09](evidence.md#c09) | [相邻权重attention变化](../docs/cache-reconstruction.md) | 已完成 局部证据 | 平均attention TV很小但存在少量大变化query行。 |
| [C10](evidence.md#c10) | [旧top100重建及新top100 oracle](../docs/cache-reconstruction.md) | 已完成 阴性结果 | 旧/新top100误差几乎相同，固定100选择损失尾部质量主导误差。 |
| [C11](evidence.md#c11) | [按累计attention质量选择](../docs/cache-reconstruction.md) | 已完成 局部证据 | 99%/99.9%改善重建质量；99.9%仍需保留约46.48%的QK对。 |
| [C12](evidence.md#c12) | [均值尾部补偿](../docs/cache-reconstruction.md) | 已完成 局部证据 | top100+均值未稳定改善续写；99.9%+均值平均误差进一步下降。 |
| [C13](evidence.md#c13) | [Qwen GQA Wv秩](../docs/cache-reconstruction.md) | 已完成 局部证据 | Wv为1024×2048且满行秩，一般E不能从V唯一恢复。 |
| [C14](evidence.md#c14) | [同权重旧KV恢复旧Q](../docs/cache-reconstruction.md) | 已完成 局部证据 | FP32联合方向约束可近似恢复，BF16缓存量化使病态逆误差很大。 |
| [C15](evidence.md#c15) | [Pythia无GQA的方阵Wv](../docs/cache-reconstruction.md) | 已完成 局部证据 | FP64阈值满秩但高条件数；没有端到端恢复Q结果。 |
| [C16](evidence.md#c16) | [K/V相邻变化的口径复核](../docs/cache-reconstruction.md) | 已完成 局部证据 | 两模型被测相邻权重的K相对变化均小于V；绝对RMS和相对误差不同。 |
| [C17](evidence.md#c17) | [Q侧变化与局部attention敏感性](../docs/cache-reconstruction.md) | 已完成 局部证据 | Q/K/V共同输入E已核对；分别替换旧Q/K/V改变attention输出；Q/K得分贡献范数不可相加。 |

## 分组采样

| 编号 | 实验与问题 | 状态 | 结果与限定 |
|---|---|---|---|
| [G-01](evidence.md#g-01) | [固定权重八同发与4+4生成微实验](../docs/group-sampling.md) | 已完成 局部证据 | 固定权重4+4耗时为八同发的1.373倍；首四全对仍可能丢掉有效G8组。 |
| [G-02](evidence.md#g-02) | [APRIL调度与pending窗口2×2短训](../docs/group-sampling.md) | 已完成 局部证据 | 扩大pending窗口可补并发，但5轮筛选不证明稳态速度或质量收益。 |
| [G-03](evidence.md#g-03) | [旧1.7B/4k八同发与4+4舍弃约20h长跑](../docs/group-sampling.md) | 已完成 局部证据 | 旧1.7B设置的4+4多提交约6.4%轮次，不能外推到8B。 |
| [G-04](evidence.md#g-04) | [8B APRIL eight、nodiscard44、discard44三组长跑](../docs/group-sampling.md) | 已完成 局部证据 | 三组输出长度和工作量不同；质量没有单一排序。 |
| [G-05](evidence.md#g-05) | [8B四种组策略的长度演化](../docs/group-sampling.md) | 已完成 局部证据 | 不同组策略都出现固定题评测长度增长，发生阶段不同。 |
| [G-06](evidence.md#g-06) | [race4 v1取消失效与排除](../docs/group-sampling.md) | 实现无效 已排除 | 目标取消未生效；首版不得用于有效取消加速的性能结论。 |
| [G-07](evidence.md#g-07) | [seqfirst串行至首个正确的可变组实验](../docs/group-sampling.md) | 核验或复核 | 串行至首次正确保留可变大小0…01组，改变了组分布。 |
| [G-08](evidence.md#g-08) | [0.6B GSM8K三策略1000轮稳定性筛选](../docs/group-sampling.md) | 已完成 局部证据 | 三组各1000轮均未触发既定失稳筛选门槛，单种子。 |
| [G-09](evidence.md#g-09) | [超过已正确短回答2倍长度时停止的离线估算](../docs/group-sampling.md) | 离线估算 | 离线2L停止能省token，也会截掉原本正确的回答。 |
| [G-10](evidence.md#g-10) | [APRIL跨轮跨度与重复前缀重建统计](../docs/group-sampling.md) | 已完成 局部证据 | 反复短续写再暂停不是该快照主流；仍有前缀重建开销。 |
| [G-11](evidence.md#g-11) | [8B正式setup迁移与生成/反向基础检查](../docs/group-sampling.md) | 已完成 局部证据 | 设置与奖励链经过基础核验；基准就绪不意味着算法更优。 |
| [G-12](evidence.md#g-12) | [Base与discard44 step503的首中次数和采样覆盖](../docs/group-sampling.md) | 已完成 局部证据 | 首答和100次覆盖提高，总生成token却增加约85.2%。 |
| [G-13](evidence.md#g-13) | [8B候选与组内rollout长度分布](../docs/group-sampling.md) | 已完成 局部证据 | 回答长度与组内最大值呈长尾，训练选中组与全部候选分母不同。 |

## 训练稳定性

| 编号 | 实验与问题 | 状态 | 结果与限定 |
|---|---|---|---|
| [S01](evidence.md#s01) | [实验设置与指标边界](../docs/training-stability.md) | 核验或复核 | early formal、late nodiscard44、full8 eight分别处理；裁剪前范数不同于Adam参数更新幅度。 |
| [S02](evidence.md#s02) | [早期首token直接损失干预](../docs/training-stability.md) | 受控干预 | 首token置零降低端点自然长度与Alright比例，随机/NLLmatched单token对照无相同效果。 |
| [S03](evidence.md#s03) | [Base与早期checkpoint的强制开头正确率和方差](../docs/training-stability.md) | 已完成 局部证据 | Base的Alright正确率高于To；G4方差差异的区间跨零，不能推出更好梯度信号。 |
| [S04](evidence.md#s04) | [实际训练中开头收益和长度惩罚的信号反转](../docs/training-stability.md) | 已完成 局部证据 | early Alright平均优势为正，late另一轨迹为负；移除长度惩罚仍负，正确率/内容也在改变。 |
| [S05](evidence.md#s05) | [Base一万条异常筛选](../docs/training-stability.md) | 已完成 局部证据 | Base已有乱码、漂移、对话与重复；单次输出异常不等于题目稳定诱发。 |
| [S06](evidence.md#s06) | [Base长度与重复、盲审误报复核](../docs/training-stability.md) | 核验或复核 | 更长回答有更高重复标签率，但因果方向未识别；盲审发现正常进展与轻度冗余被混入。 |
| [S07](evidence.md#s07) | [异常前缀、checkpoint与数据触发检查](../docs/training-stability.md) | 受控干预 | 相同失败前缀更换checkpoint强烈改变循环；局部重采样支持题目×权重×轨迹交互而非单一坏题解释。 |
| [S08](evidence.md#s08) | [屏蔽@的缓解尝试](../docs/training-stability.md) | 已完成 阴性结果 | 符号屏蔽消除@但没有恢复失败边界的正常停止或准确率，可能换为其他重复。 |
| [S09](evidence.md#s09) | [PPL尾部与时间顺序](../docs/training-stability.md) | 核验或复核 | 算术平均PPL可被极少回答主导；不能从总曲线推断单token触发或普通回答整体恶化。 |
| [S10](evidence.md#s10) | [负优势校正和token权重对应关系](../docs/training-stability.md) | 受控干预 | 负优势TIS在共同10轮抑制峰值；20轮中正确token对应的TIS显著优于均值/乱序权重。 |
| [S11](evidence.md#s11) | [最高NLL/失配token的梯度净投影](../docs/training-stability.md) | 已完成 局部证据 | 最高少量token并未解释主要净梯度，低p0/q的大集合贡献约59%。 |
| [S12](evidence.md#s12) | [固定信号下的状态响应、FP32和块回退](../docs/training-stability.md) | 受控干预 | 同一输入与固定系数的梯度随模型状态增大；FP32 master仍大；部分中间层回退降低响应但未验证修复质量。 |
| [S13](evidence.md#s13) | [full8 eight长度、开头与后期梯度](../docs/training-stability.md) | 已完成 局部证据 | 350→400长度翻倍而梯度仍小；600→650评测长度和机械重复下降但梯度上涨。 |
| [S14](evidence.md#s14) | [在线TIS/同步分支与输出质量](../docs/training-stability.md) | 在线单轨迹对照 | 从500启用TIS大幅压低梯度，但64题端点异常标签更多；低梯度不保证好质量。 |
| [S15](evidence.md#s15) | [full8 650→850续训](../docs/training-stability.md) | 运行中 | 从完整650恢复，继续200轮；首轮651验证通过，后期趋势待运行结束后分析。 |

## 历史进度

[原有实验目录](history-index.md)保留原快照。旧快照的当时运行状态不作为今天状态；研究章节按后来核对的证据更新。

<!-- final-stability-progress:start -->
## 新增研究单元

| 编号 | 实验 | 状态 |
|---|---|---|
| [S16](evidence.md#s16) | [同一600状态的八组稳定性干预](final-stability-600-650.md) | 3/8组训练、4/9个次要评测端点；候选文本覆盖见记录，尚非最终结论。 |
<!-- final-stability-progress:end -->
