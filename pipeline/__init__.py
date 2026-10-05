"""pipeline — 端到端流水线编排层。

把 data → preprocess → cs → hpo → training → eval 串成可复现的
一次运行，统一负责配置注入、随机种子传播、产物落盘与结果汇总。

作者：晨星
"""

__all__: list[str] = []
