# 聚合数据

[最新稳定性聚合数据](investigation/README.md)。

- [catalog.json](catalog.json)：原45个研究单元与新增S16，共46项及其状态。
- [evidence](evidence)：逐项结果、限制和原始归档 SHA256。
- [eight-validation.csv](eight-validation.csv)：full8 固定 1021 道 DeepMath × 4 的验证聚合。
- [eight-training.csv](eight-training.csv)：同一原始 full8 run 的训练指标。
- [first-hit.csv](first-hit.csv)：1621 题、最多 100 次首中实验的聚合覆盖。
- [legacy-manifest.json](legacy-manifest.json)：旧仓库提交及原文件 Git blob 指纹。
- [source-summary.json](source-summary.json)：附加来源与范围说明。

不同 run 的轮次不能拼成同一条训练轨迹。来源文件名及 SHA256 用于核对外部归档；本目录没有权重、原题、原回答或逐题标识。指标口径见[说明](../docs/measurement-guide.md)。

<!-- final-stability-progress:start -->
## 最后一批八组对照

[本次冻结快照](final-stability-600-650/manifest.json)纳入3/8组训练、4/9个次要评测端点。全量候选文本覆盖：0 Baseline 50/50轮；1 Full TIS 50/50轮；2 Negative TIS 3/50轮。缺失不填零，单轨迹结果与未校正区间不等于最终方法排名。Baseline的[全量候选与来源对账](final-stability-600-650/baseline_candidate_closure.json)已补齐，full TIS对账继续保留。
<!-- final-stability-progress:end -->
