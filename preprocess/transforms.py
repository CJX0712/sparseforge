"""preprocess.transforms — 送入求解器前的确定性变换。

当前提供：
- ``gram_whitener``：以 A 的 Gram 矩阵做对称白化（改善条件数；只依赖 A，无泄漏）
- ``residual_center``：对 y 去均值（测量有偏时使用）
- ``Pipeline``：按序串联上述变换并保存逆变换句柄

作者：晨星 · CJX0712
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.errors import NumericsError

__all__ = ["TransformPipeline", "gram_whitener", "residual_center"]

_EPS = 1e-12


@dataclass(slots=True)
class Whitener:
    """对称白化句柄：``A -> A W``, ``x -> W x``。"""

    W: np.ndarray
    Winv: np.ndarray

    def __post_init__(self) -> None:
        if self.W.shape[0] != self.W.shape[1]:
            raise NumericsError("白化矩阵必须方阵", shape=self.W.shape)
        if not np.all(np.isfinite(self.W)):
            raise NumericsError("白化矩阵含非有限值")


def gram_whitener(A: np.ndarray, shrink: float = 0.02) -> Whitener:
    """返回 ``W`` 使 ``(A W)^T (A W) ≈ I``。

    对 Gram 矩阵做 ``G = (1-a) G + a * tr(G)/m * I`` 收缩后 Cholesky 分解，
    保证正定。``shrink=0`` 时退化为纯 Cholesky 白化。
    """
    m = A.shape[0]
    n = A.shape[1]
    G = A.T @ A
    G = 0.5 * (G + G.T)
    a = float(np.clip(shrink, 0.0, 1.0))
    if a > 0.0:
        # G 是 n×n 的 Gram 矩阵，收缩锚点必须是同维单位阵（用 np.eye(n)，非 np.eye(m)）
        G = (1.0 - a) * G + a * (np.trace(G) / m) * np.eye(n)
    try:
        L = np.linalg.cholesky(G)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - 收缩后应恒可分解
        raise NumericsError("Gram Cholesky 失败，收缩不足", shrink=shrink) from exc
    # 白化要求 (A W)^T (A W) = I。G = L L^T，取 W = L^{-T} 使 W^T G W = I。
    # 逆变换：白化空间解 x_w 满足 A W x_w = A x，故 x = W x_w = L^{-T} x_w；
    # 流水线用 out @ Winv（= Winv^T @ out）实现，故需 Winv^T = W = L^{-T}，
    # 即 Winv = L^{-1} = inv(L)。
    W = np.linalg.inv(L).T
    Winv = np.linalg.inv(L)
    return Whitener(W=W, Winv=Winv)


def residual_center(A: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """去均值：返回 ``(A', y', mu)``，其中 ``A' = A - 1 mu^T``。"""
    mu = y.mean()
    return A - np.ones((A.shape[0], 1)) * mu, y - mu, mu


@dataclass(slots=True)
class TransformPipeline:
    """串联多个变换；每步的逆变换按逆序回放。"""

    center: bool = True
    whiten: bool = True
    shrink: float = 0.02
    _handles: list[object] = field(default_factory=list, init=False, repr=False)

    def fit_transform(self, A: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._handles = []
        A_t = np.array(A, dtype=np.float64, copy=True)
        y_t = np.array(y, dtype=np.float64, copy=True)
        if self.center:
            A_t, y_t, mu = residual_center(A_t, y_t)
            self._handles.append(("center", mu))
        if self.whiten:
            w = gram_whitener(A_t, shrink=self.shrink)
            A_t = A_t @ w.W
            self._handles.append(("whiten", w))
        return A_t, y_t

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        out = np.asarray(x, dtype=np.float64).copy()
        for kind, handle in reversed(self._handles):
            if kind == "whiten":
                out = out @ handle.Winv  # type: ignore[union-attr]
            elif kind == "center":
                out = out + handle
        return out
