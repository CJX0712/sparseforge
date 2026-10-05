"""核心数据类型：统一的求解器返回值 / 基准行 / 门禁阈值。

所有模块的接口语义统一在此定义：
- ``nmse`` 越小越好（归一化均方误差）
- ``support_f1`` 越大越好（支撑集 F1）
- ``success`` 表示 NMSE < 1e-8（数值意义上的精确恢复）

作者：晨星 · CJX0712
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "GATE_THRESHOLDS",
    "SUCCESS_NMSE",
    "SUPPORT_EPS",
    "BenchmarkRow",
    "Instance",
    "RecoveryResult",
]

#: NMSE 低于此值视为"数值精确恢复"（严格判据，仅在无噪声实例有意义）
SUCCESS_NMSE = 1e-8

#: 支撑集 F1 达到 1.0 视为"支撑集完全恢复"（success 主判据，跨噪声 regime 可用）
SUCCESS_SUPPORT_F1 = 1.0

#: 支撑集判定时忽略的最小幅度
SUPPORT_EPS = 1e-10

#: 预注册门禁阈值（方案阶段定死，禁止事后调低）
GATE_THRESHOLDS: dict[str, float] = {
    # GATE-1 旗舰 vs 弱基线（Lasso-LARS，oracle k）：aggregate NMSE 相对降低 ≥ 30%
    "g1_vs_weak_rel_gain": 0.30,
    # GATE-2 旗舰 vs 最强单法（oracle k）：aggregate NMSE 相对降低 ≥ 20%
    "g2_vs_best_rel_gain": 0.20,
    # GATE-2 显著性门槛：均值差 > k * 0.5*(std_fuse + std_base)
    "g2_sigma_factor": 0.5,
    # GATE-2 非劣门槛：旗舰 NMSE 不得差于最强单法 1.10 倍
    "g2_noninferiority_ratio": 1.10,
    # GATE-3 消融：去掉旗舰任一核心组件后 NMSE 劣化 ≥ 2%
    "g3_ablation_rel_drop": 0.02,
}


@dataclass(slots=True)
class RecoveryResult:
    """单个求解器在单个实例上的返回值。

    Attributes
    ----------
    x_hat:
        恢复的信号向量，形状 ``(n,)``。
    support:
        估计的支撑集索引，形状 ``(k_hat,)``，升序且唯一。
    residual_norm:
        ``||y - A x_hat||_2``。
    nmse:
        ``||x_hat - x_true||^2 / ||x_true||^2``，越小越好。
    support_f1:
        支撑集 F1，越大越好。
    n_iter:
        实际迭代次数。
    converged:
        是否在容差内收敛（或走完既定终止条件）。
    solver:
        求解器名。
    elapsed_sec:
        墙钟秒数（per-request，batch-of-1 口径）。
    meta:
        附加诊断信息（残差一致性分、稳定性、fallback 次数等）。
    """

    x_hat: np.ndarray
    support: np.ndarray
    residual_norm: float
    nmse: float
    support_f1: float
    n_iter: int
    converged: bool
    solver: str
    elapsed_sec: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "solver": self.solver,
            "nmse": float(self.nmse),
            "support_f1": float(self.support_f1),
            "residual_norm": float(self.residual_norm),
            "n_iter": int(self.n_iter),
            "converged": bool(self.converged),
            "elapsed_sec": float(self.elapsed_sec),
            "meta": dict(self.meta),
        }


@dataclass(slots=True)
class Instance:
    """一个压缩感知实例：``y = A x_true + noise``，``A`` 为 ``(m, n)`` 且 ``m < n``。"""

    name: str
    A: np.ndarray
    y: np.ndarray
    x_true: np.ndarray
    k_true: int
    kind: str
    snr_db: float
    seed: int
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def m(self) -> int:
        return int(self.A.shape[0])

    @property
    def n(self) -> int:
        return int(self.A.shape[1])

    @property
    def true_support(self) -> np.ndarray:
        return np.flatnonzero(np.abs(self.x_true) > SUPPORT_EPS)

    def score(self, x_hat: np.ndarray) -> tuple[float, float]:
        """返回 ``(nmse, support_f1)``。"""
        denom = float(np.sum(self.x_true**2))
        nmse = float(np.sum((x_hat - self.x_true) ** 2) / denom) if denom > 0 else 0.0
        pred = np.flatnonzero(np.abs(x_hat) > SUPPORT_EPS)
        true = self.true_support
        if pred.size == 0 and true.size == 0:
            return nmse, 1.0
        if pred.size == 0 or true.size == 0:
            return nmse, 0.0
        inter = np.intersect1d(pred, true).size
        prec = inter / pred.size
        rec = inter / true.size
        f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
        return nmse, float(f1)


@dataclass(slots=True)
class BenchmarkRow:
    """基准表的一行：某实例 × 某方法。"""

    instance: str
    seed: int
    method: str
    nmse: float
    support_f1: float
    success: bool
    residual_norm: float
    n_iter: int
    elapsed_sec: float
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance": self.instance,
            "seed": int(self.seed),
            "method": self.method,
            "nmse": float(self.nmse),
            "support_f1": float(self.support_f1),
            "success": bool(self.success),
            "residual_norm": float(self.residual_norm),
            "n_iter": int(self.n_iter),
            "elapsed_sec": float(self.elapsed_sec),
            "meta": dict(self.meta),
        }
