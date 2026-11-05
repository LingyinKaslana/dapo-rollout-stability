# 研究图表

[新报告的三组图表](../reports/rl-length-gradient-investigation.md)。

所有图由 [render_figures.py](../tools/render_figures.py) 从仓库内聚合数据生成，数据指纹见 [generated-figures.json](../data/generated-figures.json)。

| 图 | 证据与范围 |
|---|---|
| [full8 长度与后期梯度](eight-course.png) | 同一原 run；固定验证与训练指标分别展示，S13 |
| [缓存增量的 rank64 拟合](cache-rank64.png) | 1.7B 小步实验的 oracle 残差，C01、C02 |
| [三组实际工作量](group-workload.png) | 8B 前 500 轮，不是纯调度配对因果比较，G-04 |
| [首中覆盖](first-hit.png) | 1621 题、最多 100 次，G-12 |
| [首 token 干预](early-first-token.png) | 固定历史重放后的自然生成评测，S02 |
| [梯度权重对照](gradient-weight-controls.png) | 相同历史输入与起点，20 轮 80 次更新的峰值，S10 |
| [Base 长度与重复标签](base-length-repetition.png) | 一万题单次生成的描述关联，AI 辅助标签，S06 |

每幅图同时提供同名 SVG，便于缩放。重复标签经过抽样复核发现边界误判，因此图中比例不能称为人工确认的总体循环率。
