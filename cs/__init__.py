"""cs — 压缩感知（Compressed Sensing）核心算法层。

承载测量矩阵/传感算子、稀疏恢复求解器（ISTA / FISTA / CoSaMP /
OMP / ADMM 等）与恢复质量度量。本仓的 Tier-1 依赖为 numpy，
可选后端 sklearn 在此层以适配方式接入。

作者：晨星
"""

__all__: list[str] = []
