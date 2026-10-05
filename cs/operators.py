"""cs.operators — 线性算子工具：Gram 作用、谱范数、相干度、debias 重拟合。

作者：晨星 · CJX0712
"""

from __future__ import annotations

import numpy as np

from core.errors import NumericsError

__all__ = [
    "check_finite",
    "coherence",
    "debias_refit",
    "oracle_nmse",
    "soft_threshold",
    "spectral_norm",
]


def check_finite(*arrays: np.ndarray) -> None:
    for arr in arrays:
        if not np.all(np.isfinite(arr)):
            raise NumericsError("算子输出含 NaN/Inf", shape=getattr(arr, "shape", None))


def coherence(A: np.ndarray) -> float:
    """最大列间相关系数 ``max_{i!=j} |<a_i, a_j>|``（列已归一化时）。"""
    norms = np.linalg.norm(A, axis=0)
    norms = np.where(norms < 1e-12, 1.0, norms)
    An = A / norms
    G = np.abs(An.T @ An)
    np.fill_diagonal(G, 0.0)
    return float(G.max())


def spectral_norm(A: np.ndarray, n_iter: int = 30) -> float:
    """幂迭代估计谱范数 ``||A||_2``。固定初始向量 ⇒ 确定性。"""
    n = A.shape[1]
    v = np.ones(n) / np.sqrt(n)
    sigma = 0.0
    for _ in range(n_iter):
        u = A @ v
        nu = np.linalg.norm(u)
        if nu < 1e-15:
            return 0.0
        v_new = A.T @ u / nu
        nv = np.linalg.norm(v_new)
        if nv < 1e-15:
            return nu
        v = v_new / nv
        sigma = nu
    return float(sigma)


def debias_refit(A: np.ndarray, y: np.ndarray, support: np.ndarray) -> np.ndarray:
    """在给定支撑集上做最小二乘重拟合（debiasing）。

    稀疏恢复标准后处理：贪心/阈值类方法在支撑集上通常只保证"稀疏解"，
    重拟合可去掉阈值带来的幅度收缩偏差，是 NMSE 改进的主要来源之一。
    """
    idx = np.asarray(support, dtype=int)
    n = A.shape[1]
    if idx.size == 0:
        return np.zeros(n)
    if idx.min() < 0 or idx.max() >= n:
        raise NumericsError("支撑集越界", min=int(idx.min()), max=int(idx.max()))
    As = A[:, idx]
    coef, *_ = np.linalg.lstsq(As, y, rcond=None)
    out = np.zeros(n)
    out[idx] = coef
    return out


def soft_threshold(v: np.ndarray, lam: float) -> np.ndarray:
    """软阈值算子 ``sign(v) * max(|v| - lam, 0)``。满足收缩性质 ``|S(v)| <= |v|``。"""
    return np.sign(v) * np.maximum(np.abs(v) - lam, 0.0)


def oracle_nmse(x_hat: np.ndarray, x_true: np.ndarray) -> float:
    d = float(np.sum(x_true**2))
    return float(np.sum((x_hat - x_true) ** 2) / d) if d > 0 else 0.0
