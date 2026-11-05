# 聚合分析工具

[新报告重绘与验证](../data/investigation/README.md)。

`check_project.py` 用标准库检查证据编号、来源指纹、文档链接、首中覆盖单调性和图表数据指纹。

```bash
python tools/check_project.py
```

`render_figures.py` 仅使用仓库内的聚合 JSON/CSV，生成七幅 PNG 与 SVG 科学图。它不连接实验服务器、不读取原题回答、不加载模型。

```bash
python -m pip install -r requirements-analysis.txt
python tools/render_figures.py
```

数据或解释发生变化时，先更新对应证据单元，再重新绘图。重绘得到的图不代表重新运行了训练实验。

报告新增检查：`python tools/check_investigation.py`；重绘：`python tools/render_investigation_figures.py`。参考模块的5项CPU测试见[代码说明](../reference/antiloop/README.md)，它们不是模型效果证据。
