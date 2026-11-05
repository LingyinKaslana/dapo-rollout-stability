# 稳定性报告的聚合证据

这里保存52项证据主张，分早期长度（18）、后期梯度（20）、循环与干预（14）。它们与旧45项记录有重叠，不能相加当作实验数。入口为 [index.json](index.json) 和[可读实验索引](../../experiments/stability-investigation.md)。

| 文件 | 内容 |
|---|---|
| [early_length_evidence.json](early_length_evidence.json) | 不同调度、0.6B筛选、Base异常、风险题、首token干预与模式收益 |
| [late_gradient_evidence.json](late_gradient_evidence.json) | B2/B3、交叉评分、梯度分解、失配方向、精度/Adam/IS控制与反例 |
| [repetition_interventions_evidence.json](repetition_interventions_evidence.json) | 循环/边界/attention、penalty与pulse、720研究和600短训 |
| [gap_gradient.csv](gap_gradient.csv) | 651–850实际训练数据的失配和原生逐次梯度，共200轮 |
| [gap_gradient_timeline.json](gap_gradient_timeline.json) | 时间线定义、相关性与限制 |
| [high_gap_ablation.json](high_gap_ablation.json) | 固定701批、600/700权重的删除与匹配随机对照 |
| [learned_priority_endpoints.json](learned_priority_endpoints.json) | 720研究四个优先节点及题目bootstrap；不是全部中间评测已完成 |
| [preventive20.json](preventive20.json) | 600/610/620自然评测及实际辅助信号计数 |
| [manifest.json](manifest.json) | 发布文件哈希和归档源文件哈希 |
| [figures.json](figures.json) | 新图与聚合输入的哈希对应 |

`archive:` 和 `artifact_id` 表示实验存储中的来源标识，不是可下载URL。为保护运行环境，绝对路径、主机信息和进程信息已经清理；原源SHA对应归档原字节，发布SHA对应清理后的聚合文件，两者不应相等。数值源保留字段指针或key，远程完成快照另保留原始文件哈希。本次核对了93处可在本地取得的源文件哈希；重复引用可能指向同一个文件，不将其称作93次实验。

不包含权重、原始逐题回答、密钥或合成测试结果。所有计数区分题目、唯一回答、回答使用条次、token位置与token使用条次。零mask占位不计入有效辅助样本。

复核：`python tools/check_project.py`、`python tools/check_investigation.py`；重绘：`python tools/render_investigation_figures.py`。
