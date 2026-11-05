# 稳定性调查：逐项实验与证据

本索引补全从长度跃升到最新反循环训练的过程。它按问题组织证据，52条主张并非52次独立实验，也与旧45项索引有重叠。数字、配置、来源SHA和原始字段见各行的聚合证据。

[研究总报告](../reports/rl-length-gradient-investigation.md) · [旧实验索引](README.md)

## 早期长度、数据与模式选择

| 编号 | 实验/问题 | 证据类型 |
|---|---|---|
| [EL01](#el01) | 最早的问题：四种8B调度都出现快速变长，但发生轮次不同 | see evidence |
| [EL02](#el02) | 早期长度变化与后期梯度失稳需要拆开 | see evidence |
| [EL03](#el03) | 数据/模板突然改变的初步排查 | see evidence |
| [EL04](#el04) | 1xB300缩小筛选没有复现同规模的失稳 | see evidence |
| [EL05](#el05) | 历史检索：类似首词变化与一般长度脉冲不等价 | see evidence |
| [EL06](#el06) | Base1000题初筛：异常不需要RL才存在 | see evidence |
| [EL07](#el07) | 扩至10000题并全文LLM复核，补上规则漏检 | see evidence |
| [EL08](#el08) | 坏题假说的复采检验：有温和富集，未证明触发训练爆炸 | see evidence |
| [EL09](#el09) | 固定失败输入的权重干预反对单一坏题解释 | see evidence |
| [EL10](#el10) | Base长回答更常被标重复：方向与误报需要保留 | see evidence |
| [EL11](#el11) | 强制开头改变长度：生成侧可干预证据 | see evidence |
| [EL12](#el12) | 首token直接loss的固定batch重放：与删任意一token区分 | see evidence |
| [EL13](#el13) | 为什么会偏向Alright：正确率优势有证据，方差优势不确定 | see evidence |
| [EL14](#el14) | 正反馈解释与信号反转：与bandit直觉一致但不是单次中奖证明 | see evidence |
| [EL15](#el15) | 概率陡升可来自连续log-odds积累，无须先假定数值异常 | see evidence |
| [EL16](#el16) | race4筛选可能改变开头信号，但独立512题确认仍不确定 | see evidence |
| [EL17](#el17) | full8反例约束：开头、长度、重复和大梯度不是固定绑定 | see evidence |
| [EL18](#el18) | 回看早期实际数据暴露，未定位单个坏题触发点 | see evidence |

<a id="el01"></a>

### EL01 · 最早的问题：四种8B调度都出现快速变长，但发生轮次不同

长度跃升不是4+4独有；完整八路并行APRIL也出现。固定题大多数变长，不能仅用题目换了或正确/错误比例变化解释。

[数值、实验设置和来源字段](../data/investigation/early_length_evidence.json)（单元 `EL01`）。

<a id="el02"></a>

### EL02 · 早期长度变化与后期梯度失稳需要拆开

full8早期长度均值1329.6→3070.3时轮均梯度0.383→0.243，失配代理也未变大；较长回答同时伴随准确率提高。

[数值、实验设置和来源字段](../data/investigation/early_length_evidence.json)（单元 `EL02`）。

<a id="el03"></a>

### EL03 · 数据/模板突然改变的初步排查

这些验证文件没有题目或答案模板漂移；即使前后都4/4正确，full8均长1104→1974。

[数值、实验设置和来源字段](../data/investigation/early_length_evidence.json)（单元 `EL03`）。

<a id="el04"></a>

### EL04 · 1xB300缩小筛选没有复现同规模的失稳

三臂未触发预注册探索门槛；后期有缓慢变长但没有8B式明显短窗口跳变。

[数值、实验设置和来源字段](../data/investigation/early_length_evidence.json)（单元 `EL04`）。

<a id="el05"></a>

### EL05 · 历史检索：类似首词变化与一般长度脉冲不等价

seqfirst8也有100–140附近的同类首词/长度变化；旧1.7B和0.6B的脉冲较温和，TailSieve未见Alright接管且梯度未升。

[数值、实验设置和来源字段](../data/investigation/early_length_evidence.json)（单元 `EL05`）。

<a id="el06"></a>

### EL06 · Base1000题初筛：异常不需要RL才存在

至少99/1000被判异常，其中95条自然结束；异常不等价于长度截断。未命中候选之外的40条抽查又发现5条异常。

[数值、实验设置和来源字段](../data/investigation/early_length_evidence.json)（单元 `EL06`）。

<a id="el07"></a>

### EL07 · 扩至10000题并全文LLM复核，补上规则漏检

Base在这些T1采样下已有任务漂移/虚构对话/乱码/重复：2555异常、73不确定、7372normal标签；重复标签308（严格规则39）。

[数值、实验设置和来源字段](../data/investigation/early_length_evidence.json)（单元 `EL07`）。

<a id="el08"></a>

### EL08 · 坏题假说的复采检验：有温和富集，未证明触发训练爆炸

risk29.375% vscontrol25.375%，差4pp、配对95%CI[1.725,6.2]pp；≥4/8异常题20.4% vs13.8%。

[数值、实验设置和来源字段](../data/investigation/early_length_evidence.json)（单元 `EL08`）。

<a id="el09"></a>

### EL09 · 固定失败输入的权重干预反对单一坏题解释

同前缀在541的边界32/32循环、500为0/32；在选定新复采4个循环中，大loggap均在首个@后出现。支持题目×权重×路径交互。

[数值、实验设置和来源字段](../data/investigation/early_length_evidence.json)（单元 `EL09`）。

<a id="el10"></a>

### EL10 · Base长回答更常被标重复：方向与误报需要保留

AI重复标签率随长度从<1024的0.48%至4096–8192的28.39%；但循环本身会拉长回答，且8例盲审中3明显循环/3轻冗余/2正常推进。

[数值、实验设置和来源字段](../data/investigation/early_length_evidence.json)（单元 `EL10`）。

<a id="el11"></a>

### EL11 · 强制开头改变长度：生成侧可干预证据

Base强制Alright均长2748 vsTo1092（差1657，[1286,2053]token）；100轮约3203 vs1064，150仍约2200token差。单个开头足以改变条件生成长度。

[数值、实验设置和来源字段](../data/investigation/early_length_evidence.json)（单元 `EL11`）。

<a id="el12"></a>

### EL12 · 首token直接loss的固定batch重放：与删任意一token区分

首token置零把自然端点平均长度3328→2035，Alright247/256→65/256；随机和NLLmatched删一token仍3360/3271、235/241个Alright。差异不只是少训练一个token。

[数值、实验设置和来源字段](../data/investigation/early_length_evidence.json)（单元 `EL12`）。

<a id="el13"></a>

### EL13 · 为什么会偏向Alright：正确率优势有证据，方差优势不确定

Base正确率57.81% vs39.84%，差17.97pp CI[10.55,25.39]；100差8.59pp、150差9.77pp。

[数值、实验设置和来源字段](../data/investigation/early_length_evidence.json)（单元 `EL13`）。

<a id="el14"></a>

### EL14 · 正反馈解释与信号反转：与bandit直觉一致但不是单次中奖证明

early100–110 Alright尚少(169/5632)，correct68.05%、A+0.447；To45.29%、A−0.0155。与稀少高收益模式逐步更多采样和更新的正反馈相符。late501–505另一轨迹则Alright A−0.091/To+0.142，移除长度惩罚后Alright仍−0.069。

[数值、实验设置和来源字段](../data/investigation/early_length_evidence.json)（单元 `EL14`）。

<a id="el15"></a>

### EL15 · 概率陡升可来自连续log-odds积累，无须先假定数值异常

P(Alright)从100的1.97%逐步至120的15.93%、125的37.37%、130的68.57%、150的97.69%；每题logodds近线性拟合medianR²=.965，符合多步积累经sigmoid表现为陡变。

[数值、实验设置和来源字段](../data/investigation/early_length_evidence.json)（单元 `EL15`）。

<a id="el16"></a>

### EL16 · race4筛选可能改变开头信号，但独立512题确认仍不确定

native shaped reward的保留后首tokenAlright系数代理每offeredqid增加0.0270，CI[−.00618,.06263]；不能声称已证明race筛选放大首词信号。

[数值、实验设置和来源字段](../data/investigation/early_length_evidence.json)（单元 `EL16`）。

<a id="el17"></a>

### EL17 · full8反例约束：开头、长度、重复和大梯度不是固定绑定

550→600 Alright比例84.28%→90.79%但总体长度2081→1866；600→650验证长度1866→1377、严格重复6→1，而轮均grad .658→77.276。

[数值、实验设置和来源字段](../data/investigation/early_length_evidence.json)（单元 `EL17`）。

<a id="el18"></a>

### EL18 · 回看早期实际数据暴露，未定位单个坏题触发点

可以追踪已筛题的暴露，但大部分训练题未被10000题筛选覆盖，不能把unreviewed当normal；没有从这份关联中识别到原训练的确定触发题。

[数值、实验设置和来源字段](../data/investigation/early_length_evidence.json)（单元 `EL18`）。

## 后期梯度、概率失配与受控探针

| 编号 | 实验/问题 | 证据类型 |
|---|---|---|
| [LG01](#lg01) | 同一full8主线与重启边界 | provenance_and_restore_audit |
| [LG02](#lg02) | 峰值集中B2/B3，但不是第2/3次执行本身导致 | observation_plus_order_intervention |
| [LG03](#lg03) | 同样输入的反向响应随权重显著改变 | controlled_frozen_backward |
| [LG04](#lg04) | B2负优势梯度主导，范数不是可相加百分比 | controlled_sign_decomposition |
| [LG05](#lg05) | 501/601/701数据×600/700权重交叉 | controlled_fixed_input_state_comparison |
| [LG06](#lg06) | 大d token的定位与删除干预 | controlled_loss_mask_intervention |
| [LG07](#lg07) | p下降/上升×正负优势四象限 | controlled_frozen_component_intervention |
| [LG08](#lg08) | 持续重复与严重失配的强关联有边界 | observational_population_association |
| [LG09](#lg09) | 跨权重续写边界及高d token的采样概率 | aligned_segment_observation |
| [LG10](#lg10) | 零权重落后仍能严重失配 | same_weight_native_forward_counterexample |
| [LG11](#lg11) | 反例：长循环也可有很小d和很小似然梯度 | counterexample_and_fixed_objective_backward |
| [LG12](#lg12) | 实际目标审计：这里没有逐回答1/T | source_and_math_audit |
| [LG13](#lg13) | 大d不是梯度爆炸的代数充分条件 | mathematical_audit_with_counterexamples |
| [LG14](#lg14) | 精确725峰值：负token权重位置比同量缩放更重要 | verified_same_state_gradient_and_Adam_intervention |
| [LG15](#lg15) | 数值路径：同一Adam更新的非确定性及受控状态对 | controlled_numerical_intervention |
| [LG16](#lg16) | FP16并非已证实的通用修复 | supplementary_precision_counterexample |
| [LG17](#lg17) | negative_mass与固定分母：同历史短训及Adam控制 | controlled_fixed_history_training |
| [LG18](#lg18) | 真正在线20轮：原生复跑波动削弱TIS质量归因 | online_trajectory_comparison_not_replicated_treatment_effect |
| [LG19](#lg19) | 旧TIS控制与MIS讨论必须分清路线/实施状态 | supplementary_prior_branch_and_method_taxonomy |
| [LG20](#lg20) | 不能把失败保真重放当成生产机制证据 | failed_reproduction_boundary |

<a id="lg01"></a>

### LG01 · 同一full8主线与重启边界

原20261031 full8/APRIL主线从0训练；原650完整状态续跑651–850，共200轮800次更新。完整actor/Adam/scheduler/RNG与逻辑collector可恢复，但live推理RNG、KV和异步到达顺序不等价于历史原地继续。

解释边界：

- 重跑651–662与原历史重叠，不能当作新增独立实验轮数。
- 重启生成顺序/随机状态差异不止一个可由全局seed和轮数推得的计数；对照需记录实际入训数据。
- full8 baseline指未校正APRIL，不是同步DAPO。

[数值、实验设置和来源字段](../data/investigation/late_gradient_evidence.json)（单元 `LG01`）。

<a id="lg02"></a>

### LG02 · 峰值集中B2/B3，但不是第2/3次执行本身导致

800次实际更新中，所有>100的59次峰值都在B2/B3。交换执行顺序后B2第一执行仍很大；冻结轮初状态不做前序Adam也能出现大范数。批次身份与当前状态重要，顺序进一步调制幅度。

解释边界：

- native8顺序实验使用重建状态，非原生产pre726；两次重复衡量数值重复性，不是训练种子置信区间。
- B1先执行在该native8重建状态抑制B2；另一1GPU状态曾增强，方向不是普遍固定。

[数值、实验设置和来源字段](../data/investigation/late_gradient_evidence.json)（单元 `LG02`）。

<a id="lg03"></a>

### LG03 · 同样输入的反向响应随权重显著改变

相同batch726、A、mask、D，分别用实际保存720/730权重并强制own-anchor rho=1，B2/B3梯度相差约11.2/9.1倍。

解释边界：

- 这不是原始726峰值时间定位。
- 相同logp系数只排除了该系数差异；softmax残差和logits Jacobian同时可变，不能称纯Jacobian放大测量。

[数值、实验设置和来源字段](../data/investigation/late_gradient_evidence.json)（单元 `LG03`）。

<a id="lg04"></a>

### LG04 · B2负优势梯度主导，范数不是可相加百分比

batch726 B2在720/730两个真实保存状态的负优势分量几乎对齐完整梯度，正优势分量小很多。

解释边界：

- 全分母D=596129，未分别重归一化；只讨论裁剪前参数梯度。
- 闭合误差约0.13%–0.61%，不能精确解释比它更小的分量投影。
- 负优势不等同错误回答，原reward含长度惩罚。
- 尚未由该分解定位E字符本身的因果贡献。

[数值、实验设置和来源字段](../data/investigation/late_gradient_evidence.json)（单元 `LG04`）。

<a id="lg05"></a>

### LG05 · 501/601/701数据×600/700权重交叉

相同历史回答在不同权重上的d和冻结梯度显著改变；数据批次也有独立影响。实际完成的是三批数据×两份权重，不存在这里可引用的full8 500权重完整矩阵。

解释边界：

- 501只有168条/21组；601与701各512条/64组。另有matched168校验，不把全量范数当样本数匹配。
- 损失为固定A的线性似然-sum(A logp)/D、0次Adam，不是生产轮4次顺序PPO范数平均。
- q沿用历史采样，p600打分701含反事实跨权重差异；d只相关不直接输入该冻结损失。
- 使用最终aligned701语义标签，重复/乱码标签重叠且AI标签不是人工真值。

[数值、实验设置和来源字段](../data/investigation/late_gradient_evidence.json)（单元 `LG05`）。

<a id="lg06"></a>

### LG06 · 大d token的定位与删除干预

同状态保持上下文和原D，仅去掉d>0.5位置的loss系数，会大幅改变并降低冻结全批梯度；效果强于同等删量随机对照。

解释边界：

- 这是改loss支持的干预，不是直接操纵d而保持其余不变。
- 随机对照只有3次固定mask，不是3条独立训练轨迹。
- 分量可相消，部分only范数超过full；有符号投影不是正的因果贡献百分比。

[数值、实验设置和来源字段](../data/investigation/late_gradient_evidence.json)（单元 `LG06`）。

<a id="lg07"></a>

### LG07 · p下降/上升×正负优势四象限

在701批次、600和700两状态，主要大梯度来自负优势且p相对q下降的token。两方向不能只按数量或d大小合并。

解释边界：

- s=lnp0−lnq；down<−0.5，up>0.5。0.5只相当于约1.65倍概率比，未必极端异常。
- 600/700的mask按各自概率定义，位置并非固定同一集合。
- only/remove保留原A与D；固定own-anchor线性PG，不等同历史B2/B3实际PPO。

[数值、实验设置和来源字段](../data/investigation/late_gradient_evidence.json)（单元 `LG07`）。

<a id="lg08"></a>

### LG08 · 持续重复与严重失配的强关联有边界

在实际701入训512条、用700打分时，75/76重复标注回答满足严重失配定义，normal为0/413；但同一normal回答在600打分有230/413满足，表明关系依赖权重。

解释边界：

- 严重失配定义是一个回答至少5% token的d>0.5，不是任何单token超过0.5。
- 75/76重复回答达8192，而normal最长2300，无等长晚段正常对照；长度强混杂。
- 这是实际入训群体，不是全rollout。
- 短重复后恢复的row166共1773token，仅1个d>0.5；不可把任何重复都判为失配。

[数值、实验设置和来源字段](../data/investigation/late_gradient_evidence.json)（单元 `LG08`）。

<a id="lg09"></a>

### LG09 · 跨权重续写边界及高d token的采样概率

对771历史token与770 collector段落做可验证对齐后，高d尾部大多是原生成时高概率token；不是普遍由偶然低概率采样触发。边界前后均可有大d，不能把所有突变定位到续写边界。

解释边界：

- 仅已核对segments的子集；前后窗口长度/版本构成不同，均值比较不是续写边界随机干预。
- p为原round-start p0，不是B1更新后的实际B2 ptheta。

[数值、实验设置和来源字段](../data/investigation/late_gradient_evidence.json)（单元 `LG09`）。

<a id="lg10"></a>

### LG10 · 零权重落后仍能严重失配

最终740模型生成的完整token，拿同一份权重用原生训练前向重算，仍出现q≈0.98而p≈10^-7的E；权重落后不是严重失配的必要条件。

解释边界：

- cached decode与完整前向仍有精度、布局、后端、cache执行差异，尚未分离各因素份额。
- 选择了每模型8截断+8非截断；不是总体发生率。
- 这里未测这些具体token的参数梯度，不可凭d直接断言梯度爆炸。

[数值、实验设置和来源字段](../data/investigation/late_gradient_evidence.json)（单元 `LG10`）。

<a id="lg11"></a>

### LG11 · 反例：长循环也可有很小d和很小似然梯度

当前长9循环能够对坏输出高度确信、p≈q，NLL反向信号极弱；这否定“持续重复与严重失配普遍绑定”的无条件表述。

解释边界：

- NLL与固定selected-token-vs-EOS margin方向不同，norm比值不是Jacobian算子范数。
- 这些末16token反向不是实际G8 PPO全批梯度。
- 数值NLL=0不等于精确零梯度；实际全8长循环梯度极小但非零。

[数值、实验设置和来源字段](../data/investigation/late_gradient_evidence.json)（单元 `LG11`）。

<a id="lg12"></a>

### LG12 · 实际目标审计：这里没有逐回答1/T

这条运行使用global minibatch token-mean，D_j=ΣT；同组A广播至response token。不是原始sequence-mean GRPO的每回答1/T。长回答会改变全局分母与总系数质量，不能直接套逐回答惩罚被稀释的解释。

解释边界：

- 替换固定D在一个固定batch只缩放整批向量，跨更新的Adam、epsilon、clip、moment和数据变化使轨迹效果不同。
- 全局token-mean仍是随机分母ratio estimator，不能称完全长度无偏。

[数值、实验设置和来源字段](../data/investigation/late_gradient_evidence.json)（单元 `LG12`）。

<a id="lg13"></a>

### LG13 · 大d不是梯度爆炸的代数充分条件

原生未做rollout correction时q不是PPO loss显式输入；rho使用p/p0。own-anchor rho=1，即使p0/q极小，logp系数仍由A/D决定。梯度还取决于softmax残差、参数Jacobian、方向叠加及裁剪。

解释边界：

- q决定采样分布及哪些历史轨迹进入数据，不代表整体训练里无关。
- p很小时1−p饱和，d从50增至100不把输出残差翻倍。
- 尚无普适的必要充分条件或precision/staleness/repetition因果百分比分解。

[数值、实验设置和来源字段](../data/investigation/late_gradient_evidence.json)（单元 `LG13`）。

<a id="lg14"></a>

### LG14 · 精确725峰值：负token权重位置比同量缩放更重要

新增受控固定历史轨迹实际725 B2峰值被完整保真复现。保持同状态，negative-TIS比匹配负系数总量的统一缩小更有效压低当前梯度；一步Adam后native原锚点梯度下降但线性梯度上升，刷新锚点又恢复大梯度。

解释边界：

- This is a verified new controlled trajectory actual peak, not a recovered original771 state.
- Frozen coefficient weights identify a same-state loss intervention, not long-run generated quality.
- PPO-bounds-removed branches retain native log-ratio clamp[-20,20]; not unbounded exponentials.
- Negative and positive vector norms are not additive fractions; measured closure is approximate.
- Post-native vs refreshed-anchor contrast includes ratio and clipping effects; it does not isolate one clip bound.
- TIS uses own-round anchor vs historicalq, frozen weights; no claim a generic online TIS scheme improves quality.

[数值、实验设置和来源字段](../data/investigation/late_gradient_evidence.json)（单元 `LG14`）。

<a id="lg15"></a>

### LG15 · 数值路径：同一Adam更新的非确定性及受控状态对

相同完整730状态重复一次B1：非确定性FA2反向向量差约1.2%；确定性FA2后3次gradient/Adam/post-logp逐位相同。但确定性训练仍可出现峰值。两近邻实际poststate的BF16路径可放大极小参数差，math attention也能出现，FA2不是必要条件。

解释边界：

- poststates来自受控单B1结果，不是原生产后继权重。
- row150罕见换行logp差67–75nats不等同整个词表分布同等幅度改变；两模型仍可都确信9。
- FP32是反事实计算参考，不是“正确策略”或已证实的训练修复。

[数值、实验设置和来源字段](../data/investigation/late_gradient_evidence.json)（单元 `LG15`）。

<a id="lg16"></a>

### LG16 · FP16并非已证实的通用修复

辅助late nodiscard44的FP16探测包含非有限反向；改变loss scale后部分梯度稳定，但不能推出FP16训练整体稳定或优于BF16。

解释边界：

- native517/natural541属于另一条late分支，不把编号混成full8主线。
- 同前向logp、loss scale128时natural541_0087有非有限参数梯度；scale1/16有限也只是预选案例。

[数值、实验设置和来源字段](../data/investigation/late_gradient_evidence.json)（单元 `LG16`）。

<a id="lg17"></a>

### LG17 · negative_mass与固定分母：同历史短训及Adam控制

720完整状态上721–730固定历史数据四臂，分别保持正/零A并调整负回答长度分配，或固定D。负系数总量匹配后位置仍变；真实Adam位移不按裁剪前范数比例变化。

解释边界：

- 这是固定历史数据训练，不是在线采集；own-p0每轮重算，各臂后续状态不同。
- negative_sequence是按1/T重新分配负质量并匹配总量，不是把整套GRPO目标换成sequence-mean。
- negative_cap2048缩放长负回答的总权重，不是裁掉2048之后上下文。
- 均值/最大裁剪前范数不代表Adam参数位移；总step位移L2之和也非起终点净距离。
- 同norm原生方向注入与真实加权方向的poststate仍不同；fixedD方向理论只缩放，BF16实测留有小数值残差。

[数值、实验设置和来源字段](../data/investigation/late_gradient_evidence.json)（单元 `LG17`）。

<a id="lg18"></a>

### LG18 · 真正在线20轮：原生复跑波动削弱TIS质量归因

同720起点两条原生轨迹的准确率差14.84个百分点；TIS相对第一次原生看似提升16.21点，相对第二次只有1.37点且题级区间跨零。固定D这次未见改善。

解释边界：

- All comparisons condition on trained weights and reused128qid panel; qid-bootstrap intervals do not include training trajectory variability.
- Only two native nominal-setting/runtime trajectories, and one TIS trajectory; no general treatment effect or training-seed distribution is established.
- Intervals are exploratory and unadjusted across multiple comparisons.
- No G8/advantages are reconstructed from four fresh answers.
- Strict rule-negative does not mean normal language.
- 原生/fixedD首轮入训数据在首次更新前已经不同；不是只改分母的同数据在线随机对照。
- 严格规则排除数字，strict=0时仍可能全部截断回答都是长9；需看周期/截断和原文。

[数值、实验设置和来源字段](../data/investigation/late_gradient_evidence.json)（单元 `LG18`）。

<a id="lg19"></a>

### LG19 · 旧TIS控制与MIS讨论必须分清路线/实施状态

另一条late nodiscard44在500→520/550做过full/negative-only/positive-onlyTIS、统一均值与位置打乱控制；支持权重对应位置重要，但不能并入full8轨迹或外推普遍质量收益。MIS是讨论/文献中的不同mask策略；本次已审源没有独立MIS在线效果证据。

解释边界：

- TIS权重detach(min(exp(clamp(logp0−logq,−20,20)),2))，clamp带exp(−20)数值下限，极端处不等同无clamp的p0/q。
- negative-only TIS保持正token权重1；非对称操作、group标准化与mixed筛选下不是无偏完整轨迹IS。
- TIS保留被upper-cap截住的token；MIS常以mask拒绝超界token或sequence，粒度/上下界依论文与代码，不可只凭名字等同。
- 完整序列/prefix分布校正不同于逐token的局部ratio修正。

[数值、实验设置和来源字段](../data/investigation/late_gradient_evidence.json)（单元 `LG19`）。

<a id="lg20"></a>

### LG20 · 不能把失败保真重放当成生产机制证据

original771实际B2约4.07，但一些冻结/重建测量很大；对应原Adam状态及轨迹保真未通过的尝试被保留为失败，不冒称重现原始生产峰值。后来另选实际725完整状态做精确干预。

解释边界：

- “当时没有拿到可验证的原Adam”是该次attempt的可用性记录，不宣称后来所有备份永远不存在。
- own-anchor冻结梯度与实际post-B1/clipped B2不是同一目标。
- 校准一批p0或第一步范数一致，不保证后面整条数值轨迹相同。

[数值、实验设置和来源字段](../data/investigation/late_gradient_evidence.json)（单元 `LG20`）。

## 循环机制与反重复干预

| 编号 | 实验/问题 | 证据类型 |
|---|---|---|
| [R01](#r01) | 持续重复与严重失配的关联很强，但具有模型状态条件 | observational |
| [R02](#r02) | 权重零落后也能出现大的局部d；循环末端比开端更富集 | observational |
| [R03](#r03) | 大d并非主要来自低概率偶然token | observational |
| [R04](#r04) | 跨权重续写边界不普遍触发d上跳 | observational |
| [R05](#r05) | 注意力双端偏重不等于循环专属的V形因果机制 | observational |
| [R06](#r06) | 稳定循环后注意力不主要停留在最初异常过渡区 | observational |
| [I01](#i01) | 全局 repetition penalty 试验失败，且RP1.05有后端混淆 | intervention_with_partial_confound |
| [I02](#i02) | 单目标token的短暂退出pulse没有真正逃离循环 | paired_decoding_intervention |
| [I03](#i03) | 冻结梯度下降不等于循环被修复，d也未同步改善 | frozen_gradient_diagnostic |
| [L01](#l01) | 720 E-only learned UL 已完成五阶段合计50轮，而非单条轨迹50轮 | completed_online_training |
| [L02](#l02) | 720试验没有显示主梯度峰值被解决；后10轮固定E-bank几乎已无有效信号 | online_gradient_observation |
| [L03](#l03) | 四个优先自然评估节点未支持E-only方案有效 | completed_priority_evaluation |
| [P01](#p01) | 600→620通用在线bank短跑已完成，并实际产生过辅助梯度 | completed_short_online_training |
| [P02](#p02) | 600短跑的自然评估反而出现更多循环，不能称成功预防 | completed_short_native_evaluation |

<a id="r01"></a>

### R01 · 持续重复与严重失配的关联很强，但具有模型状态条件

实际入训 batch701 在 theta700 上，重复标签组75/76达到严重失配阈值，明确正常组0/413；在theta600上，同一正常组230/413也达到阈值。因此不能泛化为“只有重复才失配”或“文本标签独自决定失配”。

解释边界：

- 512条实际入训回答，不是全部rollout；严重定义为一条回答≥5% token 的 d>0.5。
- 旧标签用于统计分组，shared prompt依赖、长度触顶与重复高度共线；没有长度8192的充分正常对照。
- first512与固定位置窗分析缓解部分长度混淆，但不是因果随机干预。
- 短重复后恢复的row166只有1/1773个token d>0.5，是“有重复必严重失配”的反例。

[数值、实验设置和来源字段](../data/investigation/repetition_interventions_evidence.json)（单元 `R01`）。

<a id="r02"></a>

### R02 · 权重零落后也能出现大的局部d；循环末端比开端更富集

两组同权重oracle共选32条回答。native740的极端d既存在于重复内部，也存在于周期变化/结束前；末16 token窗口平均d0.2416，高于内部0.0515和起始16 token0.0780。不能把所有大d定位为首次进入重复的一瞬。

解释边界：

- 每个模型选8条触顶+8条非触顶，非随机发生率估计。
- 这里比较同一权重下不同推理/训练执行路径，排除了版本落后必要性，没有排除数值精度、上下文长度、执行路径等因素。
- 检测的是精确token周期；异常语义/字符别名可能早于或跨过此边界。
- “末端”只指有实际break的末16，触顶截尾不当break。

[数值、实验设置和来源字段](../data/investigation/repetition_interventions_evidence.json)（单元 `R02`）。

<a id="r03"></a>

### R03 · 大d并非主要来自低概率偶然token

零落后样本的 d>5 共218个token，全部p<q，q中位数0.9046，179/218的q≥0.5；实际batch771 B2的旧p0-vs-q大差异token更常有接近1的历史q。大量异常续写在给定其异常前缀后是生成器的高概率行为。

解释边界：

- q是给定精确自回归前缀时实际采中token的概率，不是整条异常模式、题目或最终答案的概率。
- 后半段q高不能推断异常起点也高概率。
- 771的p是原theta770轮初p0，不是经过B1之后的实际B2训练概率。

[数值、实验设置和来源字段](../data/investigation/repetition_interventions_evidence.json)（单元 `R03`）。

<a id="r04"></a>

### R04 · 跨权重续写边界不普遍触发d上跳

batch771可精确匹配478/512条、247条含326个版本切换边界。边界前后16-token窗，103次d增加、223次减少，变化中位数-0.01547；64/128窗也多数下降。跨版本影响存在，但“重建/续写边界必然制造突发失配”不成立。

解释边界：

- Saved segment metadata checks contiguous token exact equality and q byte equality against full saved collector prefix, itself matched by prompt+UID+response prefix+q bytes; only unique matches admitted.
- 83 remaining suffixes have inferred version770 using complete saved prefix, unfinished prior state, batch max_version770, saved versions<=769, and single-version transition setup. Their provenance explicitly differs from saved collector_segment; no per-token saved historical segment exists for this suffix.
- 34 unmatched rows remain outside matched-token/boundary analyses. 32 report min=max770; 2 report769->770 but precise split unavailable. Do not classify all unmatched as no cross-version.
- q is probability of sampled token given exact autoregressive prefix under that segment generation weights, not probability of problem/correct answer/full sequence. High q on later repeated tokens does not mean beginning of abnormal mode was probable.
- p0 is recorded theta770 round-start probability before step771 actor updates. This is not actual B2 probability after B1; not a direct measurement of actual B2 gradient.
- Event-pooled windows can overlap on short neighboring segments; events/tokens are not independent. Association at real version boundary does not identify versionlag vs input/model/execution causes.

[数值、实验设置和来源字段](../data/investigation/repetition_interventions_evidence.json)（单元 `R04`）。

<a id="r05"></a>

### R05 · 注意力双端偏重不等于循环专属的V形因果机制

同权重native2_740的8条异常回答+8条对照，用HF BF16 teacher forcing回放全部36层×32头。异常起点附近recent128质量在8/8例中上升，均值0.3768→0.4966→0.6170；但“首16和末128每key密度高于中部”的双端指标在正常对照也很普遍。

解释边界：

- 这是后验HF回放的attention，不是历史vLLM生成时存下的attention。
- 指标是双端密度比较，不是拟合的字母V曲线，也不是预注册假设检验。
- 没有attention干预，因此不能认定它导致重复或d/梯度峰值。
- 36×32=1152个头全部保留；具体illustrative head为后验选择。

[数值、实验设置和来源字段](../data/investigation/repetition_interventions_evidence.json)（单元 `R05`）。

<a id="r06"></a>

### R06 · 稳定循环后注意力不主要停留在最初异常过渡区

固定“异常开始后的128个旧key”平均attention质量由刚过渡时0.6204，降至稳定EES循环0.0463、末段0.0241；后期更大质量分布在累积循环历史、最近128 token与序列起始sink。个别头能持续关注过渡区，不代表总体注意力主要在那里。

解释边界：

- attention_location每阶段使用已保存的中心query；与跨8个query均值的另一个汇总口径有小幅数值区别，不混用。
- 质量总量与每key密度须区分：大量旧key的总质量可以很高，但每个旧key的密度低。
- 按头结果由256 bins给上下界并受FP16存储舍入影响；不是精确头内区间和。

[数值、实验设置和来源字段](../data/investigation/repetition_interventions_evidence.json)（单元 `R06`）。

<a id="i01"></a>

### I01 · 全局 repetition penalty 试验失败，且RP1.05有后端混淆

source720固定权重三arm、各512回答：baseline正确250、持续重复118；RP1.05正确91、重复414；RP1.10正确2、重复511。均未实现持续退化抑制。后验发现RP1.05因网络探测超时切换到native FlashInfer，而baseline/RP1.10使用TRTLLM，故RP1.05不能隔离penalty因果效应。

解释边界：

- 纯固定权重生成+冻结反向诊断，无online optimizer更新。
- RP1.10与baseline在后端维度相同，但单面板/单试验不能推断所有强度都失败。
- RP1.10的97.0527% token处于own-A=0，own-A梯度小可能只是全错/全对组不产生GRPO信号。
- raw q与penalty后的behavior q必须分开；后者含主动解码变换。
- 冻结730是fixed-history反事实权重，非实际生产730。

[数值、实验设置和来源字段](../data/investigation/repetition_interventions_evidence.json)（单元 `I01`）。

<a id="i02"></a>

### I02 · 单目标token的短暂退出pulse没有真正逃离循环

118个已确认E循环前缀，配对native vs 最多8步将预测循环token概率压至0.5、采到不同token即关闭的一次pulse。116条在8步内换token，但88条只换成文本别名，28条有字符离开后又循环。两arm均118/118继续持续重复并触顶，0自然结束、0新增答对。

解释边界：

- 条件是已经陷入循环的118前缀，不代表完整dev64总体正确率；两条最终correct来自前缀中已存在答案。
- tokenID变化、decoded文字离开、恢复有意义数学是三种不同事件。
- 只评估新continuation，不能因固定旧prefix已经有128E直接判失败。
- 没有语义LLM复核；字符覆盖率不等于每个残余片段已被语义判断。

[数值、实验设置和来源字段](../data/investigation/repetition_interventions_evidence.json)（单元 `I02`）。

<a id="i03"></a>

### I03 · 冻结梯度下降不等于循环被修复，d也未同步改善

targeted pulse的固定负系数诊断梯度在720从1.5250降到0.3632，反事实730从5433.27降到3756.59；但重复保持118/118，且post8 raw d超0.5条件均值在730从54.3383升到58.6531。这一试验没有成功操纵“减少重复”中介，无法证明减少重复→减少d→改善长期训练的因果链。

解释边界：

- 全部10次backward含复现控制，optimizer steps为0。
- full/common在本数据支持完全一致，不是两份独立证据。
- 完整梯度仍包括最初8步，不能把其下降归因于post8部分。
- 固定负系数是诊断反事实，不能直接作为正常训练目标。

[数值、实验设置和来源字段](../data/investigation/repetition_interventions_evidence.json)（单元 `I03`）。

<a id="l01"></a>

### L01 · 720 E-only learned UL 已完成五阶段合计50轮，而非单条轨迹50轮

baseline720→730→740共20轮；UL720→730共10轮，随后从同一完整UL730分别继续UL10轮或撤掉UL10轮；合计50执行轮、200主Adam更新。正常APRIL/RL目标与decoder保留，辅助bank固定56条/24个训练题，实际唯一family是E及18种token别名。

解释边界：

- 这是720之后E-only固定bank学习，不是600之后通用在线bank。
- 五阶段完整model/Adam/scheduler/RNG/逻辑APRIL池恢复，但live vLLM RNG/KV及异步未来调度不逐位复现。
- 两条730分支同源且共同包含前10轮UL，不可当相互独立训练重复。
- 5%限制的是当前合成梯度分量范数，不是Adam参数更新的5%上界。

[数值、实验设置和来源字段](../data/investigation/repetition_interventions_evidence.json)（单元 `L01`）。

<a id="l02"></a>

### L02 · 720试验没有显示主梯度峰值被解决；后10轮固定E-bank几乎已无有效信号

721–730的RL-only最大值baseline740.27、UL1153.69；731–740三分支均降到个位数峰值，baseline3.344、ULkeep2.177、withdraw1.937。ULkeep后10轮的有效UL分量最大仅1.94e−24，因此其低峰值不能简单归因于正在持续施加强UL。

解释边界：

- 每arm仅一条异步训练轨迹，无多seed；不能把最大值差异当稳定处理效应。
- 必须使用RL-only范数作为主指标，combined可在单步直接被辅助梯度改变。
- 后期固定E-prefix上的pbad或UL梯度趋近零不证明自然生成的所有退化已经消失。

[数值、实验设置和来源字段](../data/investigation/repetition_interventions_evidence.json)（单元 `L02`）。

<a id="l03"></a>

### L03 · 四个优先自然评估节点未支持E-only方案有效

同一dev64×8、正常无penalty采样：source720准确率53.32%、总cycle22.07%；baseline740准确率55.86%、总cycle3.71%；ULkeep740准确率50.78%、总cycle17.97%；ULwithdraw740准确率52.15%、总cycle9.375%。三个740节点字母strict均0，但仍分别19/92/48条数字循环。

解释边界：

- 字母strict=0不是总重复=0；baseline同样消除了该字母标签，不能把字母消失归功于UL。
- 64题cluster bootstrap只描述该面板采样的不确定性，不包含训练seed/异步轨迹不确定性；重复使用的dev不是未触碰测试集。
- ULkeep-baseline准确率差−5.078pp、总cycle差+14.258pp为观测处理对比；不要泛化为“所有UL必然有害”。
- 总评估计划11节点，当前其他节点仍在推进；这四节点结果完整，但不声称全部11节点已完成。

[数值、实验设置和来源字段](../data/investigation/repetition_interventions_evidence.json)（单元 `L03`）。

<a id="p01"></a>

### P01 · 600→620通用在线bank短跑已完成，并实际产生过辅助梯度

最新只读远程确认20轮、80次更新、原checkpoint不变且清理完毕。与720方案不同，bank从当前训练过滤前完成组刷新，含全错组；严格排除dev，按题轮转，缺位zero-mask。实际303次非零轨迹使用来自76条回答/73题，共9032目标token，48/80次更新有效UL非零。

解释边界：

- 单UL arm，没有新匹配baseline；最初601–607有效UL为0，首次608出现。
- 使用次数≠唯一轨迹/题数；零mask占位不能算UL训练样本。
- 设计支持字母/数字/符号，但实际本次使用302个alphabetic槽、1个other槽、0个numeric槽；不能声称已在真实数字循环上验证UL。
- 128是每轮预筛候选扫描上限，不是全部rollout；漏检/覆盖率需单独报告。
- 训练过滤前采集与文本检查只在暂停后处理，不改变原RL奖励/优势/分组过滤；时序CPU开销仍可能影响异步后续轨迹。

[数值、实验设置和来源字段](../data/investigation/repetition_interventions_evidence.json)（单元 `P01`）。

<a id="p02"></a>

### P02 · 600短跑的自然评估反而出现更多循环，不能称成功预防

dev64×8在600/610/620：准确率51.1719%→49.6094%→44.5313%；cycle候选0→43→117/512；平均长度2433.45→2133.43→3043.12；触顶5→48→130/512。620中116字母循环、1符号循环、0数字循环。该单arm短跑证明工程/信号覆盖可运行，未证明减轻退化或避免未来峰值。

解释边界：

- 600的0是512条特定dev采样未命中机械规则，不能称source600模型/训练流无重复。
- 没有同步600→620无UL对照，因此既不能说UL造成全部恶化，也不能说它已经抑制了反事实更坏轨迹。
- 20轮没覆盖原计划640–650/更后期，不能判定长期预防/反弹。
- cycle是保守机械候选，不等于逐条语义乱码判决。
- native600/610/620和旧720评估虽同题面板，推理拓扑/随机流等不同；主对比应保持同一路径内部，不无条件拼成一条因果曲线。

[数值、实验设置和来源字段](../data/investigation/repetition_interventions_evidence.json)（单元 `P02`）。
