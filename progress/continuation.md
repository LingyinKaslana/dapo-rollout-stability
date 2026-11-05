# full8 650→850续训：已完成

从完整650恢复的新分支已完成200轮、800次优化器更新。逐轮失配和原生梯度可在[聚合CSV](../data/investigation/gap_gradient.csv)中复核。

恢复包括actor/Adam/调度器/训练RNG/逻辑续写池，但未精确恢复活跃vLLM随机流、KV与异步完成顺序。它不是原651–662历史的逐样本重演。

[最新总报告](../reports/rl-length-gradient-investigation.md) · [完成快照](continuation.json) · [保留的启动阶段快照](continuation-initial-snapshot.json)
