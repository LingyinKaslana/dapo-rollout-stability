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

[final-stability-600-650](final-stability-600-650/manifest.json)按已完成组增量发布。当前3/8训练完成、次要评测4/9；全量候选文本统计为baseline27/50、full TIS50/50、negative TIS0/50。训练聚合CSV保留原SHA，图为标有partial的CSV重绘；发布清单记录摘要转换。Full TIS的[候选来源对账](final-stability-600-650/full_tis_candidate_closure.json)单独列出，局部complete不代表八组研究完成。
<!-- final-stability-progress:end -->
