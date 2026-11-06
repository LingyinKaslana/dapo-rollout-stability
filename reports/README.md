# 研究报告

[从输出长度跃升到梯度异常](rl-length-gradient-investigation.md) 是当前总报告，覆盖实验路线、关键数值、机制干预、负结果与结论修正。

[逐项实验记录](../experiments/stability-investigation.md)提供52项带来源的主张，不能将该数量当成独立实验或训练种子数量。

<!-- final-stability-progress:start -->
## 按组补充的稳定性实验

[本次冻结快照](rl-length-gradient-investigation.md#final-stability-600-650)纳入4/8组训练、5/9个次要评测端点。全量候选文本覆盖：0 Baseline 50/50轮；1 Full TIS 50/50轮；2 Negative TIS 50/50轮；3 Absolute-gap mask 50/50轮。已纳入四组的候选均为全50轮；其余四组尚未纳入，整个八组实验仍未完成。单轨迹结果与未校正区间不等于最终方法排名。
<!-- final-stability-progress:end -->

<!-- sixth-native-notice:start -->
原生训练已有6/8组完成，后两组尚未完成；第五、第六组次要评测与全量候选统计仍待补。见[第六组严格重复位置覆盖及六组训练曲线](rl-length-gradient-investigation.md#sixth-native-supplement)。前四组完整比较快照和五个次要端点保持原范围。
<!-- sixth-native-notice:end -->
