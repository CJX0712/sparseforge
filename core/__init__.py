"""core — SparseForge 核心基础设施层。

提供全局复用的基础件：随机数与可复现性控制、配置对象、日志、
数值工具与类型定义。本层不依赖任何具体算法实现，供 data / cs /
preprocess / hpo / training / eval / pipeline 各层共同引用。

作者：晨星
"""

__all__: list[str] = []
