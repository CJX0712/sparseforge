"""cs.reweighted — 重加权 ℓ1（IRL1）及其与 AMP 的融合。

理论依据
--------
Candès, Wakin & Boyd, *Enhancing Sparsity by Reweighted l1 Minimization*
(J. Fourier Anal. Appl. 2008)：迭代重加权

.. math::  w_i^{(j)} = 1 / (|x_i^{(j)}| + \\varepsilon)

把 ℓ1 换成加权 ℓ1 ``sum_i w_i |x_i|`` 反复求解，会**二次方地**逼近 ℓ0 约束，
在相变区（phase transition）严格优于单次 ℓ1。论文给出的是**渐近**保证，
本系统以实测增益验证。

本模块提供三个层次（都可单独消融）：
    irl1_omp   OMP 支撑集 → 重加权 → OMP 支撑 → 迭代（启发式 IRL1）
    reweighted_fista  重加权 FISTA（L1 外层 × FISTA 内层）
    amp_rl1    AMP 内层 + 重加权外层（理论上最强的组合）

作者：晨星 · CJX0712
"""

from __future__ import annotations

import time

import numpy as np

from core.config import SolveConfig
from core.types import RecoveryResult
from cs.operators import debias_refit, soft_threshold, spectral_norm
from cs.solvers import _default_lam, _finish, _support_of

__all__ = ["amp_rl1", "irl1_omp", "reweighted_fista"]


def _fista_inner(
    A: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float, max_iter: int, tol: float
) -> np.ndarray:
    """加权 LASSO 的 FISTA 内层：``min 1/2||Ax-y||^2 + lam * sum_i w_i |x_i|``。"""
    n = A.shape[1]
    step = 1.0 / max(spectral_norm(A) ** 2, 1e-12)
    x = np.zeros(n)
    z = x.copy()
    t = 1.0
    for _ in range(max_iter):
        grad = A.T @ (A @ z - y)
        x_new = soft_threshold(z - step * grad, lam * w)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        z_new = x_new + ((t - 1.0) / t_new) * (x_new - x)
        x, z, t = x_new, z_new, t_new
        if np.linalg.norm(x_new - z) <= tol * max(1.0, np.linalg.norm(x_new)):
            break
    return x


def reweighted_fista(
    A: np.ndarray,
    y: np.ndarray,
    k: int | None = None,
    cfg: SolveConfig | None = None,
    outer: int = 4,
    eps_rel: float = 0.05,
    lam_scale: float = 1.0,
) -> RecoveryResult:
    """重加权 FISTA：Candes-Wakin-Boyd IRL1 包裹 FISTA 内层。

    Parameters
    ----------
    outer:
        重加权外层迭代数。
    eps_rel:
        稳定化常数，取 ``eps_rel * max|x|``；过小会数值爆炸。
    lam_scale:
        正则强度整体缩放。
    """
    cfg = cfg or SolveConfig()
    t0 = time.perf_counter()
    n = A.shape[1]
    lam = lam_scale * _default_lam(A, y, cfg)
    x = np.zeros(n)
    w = np.ones(n)
    for _ in range(max(1, outer)):
        x = _fista_inner(A, y, w, lam, cfg.max_iter, cfg.tol)
        mx = float(np.max(np.abs(x)))
        if mx <= 0.0:
            break
        w = 1.0 / (np.abs(x) + eps_rel * mx)
        w = w / float(np.max(w))  # 归一化，避免 lam 尺度漂移
    if cfg.debias:
        sup = _support_of(x)
        if sup.size:
            x = debias_refit(A, y, sup)
    sup = _support_of(x)
    return _finish(
        x,
        A,
        y,
        k,
        cfg,
        "reweighted_fista",
        t0,
        max(1, outer),
        True,
        debias=False,
        meta={"lam": lam, "outer": outer, "n_nonzero": int(sup.size)},
    )


def irl1_omp(
    A: np.ndarray,
    y: np.ndarray,
    k: int | None = None,
    cfg: SolveConfig | None = None,
    outer: int = 3,
) -> RecoveryResult:
    """IRL1-OMP：重加权支撑投票。

    外层维护权重 ``w_i = sum_{j} 1{第 j 轮选中 i}``，内层用 ``w`` 加权相关度
    ``|a_i^T r| * w_i`` 选支撑，再 LS 重拟合。权重随轮次单调集中到真支撑。
    """
    cfg = cfg or SolveConfig()
    t0 = time.perf_counter()
    n = A.shape[1]
    kk = cfg.k if k is None else k
    if kk is None:
        kk = max(1, n // 20)
    kk = int(min(max(1, kk), n))
    w = np.ones(n)
    sup = np.zeros(0, dtype=int)
    x = np.zeros(n)
    for _ in range(max(1, outer)):
        r = y - A @ x
        score = np.abs(A.T @ r) * w
        if sup.size:
            score[sup] += np.inf  # 已选入的保留
        cand = np.argpartition(score, -kk)[-kk:]
        sup = cand[np.argsort(-score[cand])]
        xs = np.zeros(n)
        coef, *_ = np.linalg.lstsq(A[:, sup], y, rcond=None)
        xs[sup] = coef
        x = xs
        w = 1.0 / (np.abs(x) + 1e-3 * max(float(np.max(np.abs(x))), 1e-12))
        w = w / float(np.max(w))
    if cfg.debias:
        x2 = debias_refit(A, y, sup)
        x = x2
    return _finish(
        x,
        A,
        y,
        kk,
        cfg,
        "irl1_omp",
        t0,
        max(1, outer),
        True,
        debias=False,
        meta={"k": kk, "outer": outer},
    )


def amp_rl1(
    A: np.ndarray,
    y: np.ndarray,
    k: int | None = None,
    cfg: SolveConfig | None = None,
    outer: int = 3,
    eps_rel: float = 0.05,
) -> RecoveryResult:
    """重加权 AMP：AMP 内层（含 Onsager 修正）+ IRL1 外层。"""
    cfg = cfg or SolveConfig()
    t0 = time.perf_counter()
    n = A.shape[1]
    lam = _default_lam(A, y, cfg)
    alpha = float(np.clip(cfg.damping, 0.0, 1.0))
    aty = A.T @ y
    w = np.ones(n)
    x = np.zeros(n)
    t = np.zeros(n)
    ynorm = max(float(np.linalg.norm(y)), 1e-12)
    it_tot = 0
    for _ in range(max(1, outer)):
        for _ in range(cfg.max_iter):
            it_tot += 1
            r = y - A @ x
            z = A.T @ r + t
            x_new = soft_threshold(z, lam * w)
            t_new = x_new - soft_threshold(A.T @ r - aty, lam * w)
            if alpha < 1.0:
                x_new = alpha * x_new + (1.0 - alpha) * x
            if not np.all(np.isfinite(x_new)) or float(np.linalg.norm(x_new)) > 50.0 * ynorm:
                raise RuntimeError("amp_rl1 diverged")
            d = float(np.linalg.norm(x_new - x))
            x = x_new
            t = t_new
            if d <= cfg.tol * max(1.0, float(np.linalg.norm(x))):
                break
        mx = float(np.max(np.abs(x)))
        if mx <= 0.0:
            break
        w = 1.0 / (np.abs(x) + eps_rel * mx)
        w = w / float(np.max(w))
    if cfg.debias:
        sup = _support_of(x)
        if sup.size:
            x = debias_refit(A, y, sup)
    return _finish(
        x,
        A,
        y,
        k,
        cfg,
        "amp_rl1",
        t0,
        it_tot,
        True,
        debias=False,
        meta={"lam": lam, "outer": outer, "onsager": True},
    )
