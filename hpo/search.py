"""hpo.search — 真值无关的超参搜索（只用训练折的 CV 残差）。

防泄漏硬约束：**不接收 x_true / k_true**。搜索目标恒为行交叉验证残差。
作者：晨星 · CJX0712
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from core.config import SolveConfig
from cs.solvers import BASELINES, SOLVERS

__all__ = ["grid_search_ratio", "search_all_baselines"]


def grid_search_ratio(
    solve_fn: Callable[..., object],
    A: np.ndarray,
    y: np.ndarray,
    k: int,
    ratio_grid: tuple[float, ...],
    **kwargs: object,
) -> tuple[float, float]:
    """在 ``ratio_grid``（相对 ``max|A^T y|`` 的比例）上搜索最优正则强度。

    返回 ``(best_ratio, best_score)``。评分口径 = **不一致性原理**：
    ``| ||r||^2 - m * sigma_hat^2 |``，其中 ``sigma_hat`` 由 ``cs.kselect``
    的真值无关估计器给出。

    **为什么不用 CV 残差**（实测）：CV 残差随 k 单调下降，作为 lam 的评分
    会系统性偏向过拟合解。**为什么不用训练残差**：``m < n`` 时训练残差是
    有偏信号，局部搜索实测比最优单法差 +229%。
    """
    from cs.kselect import estimate_sigma

    sh = estimate_sigma(A, y)
    m = A.shape[0]
    target = m * sh * sh
    aty = float(np.max(np.abs(A.T @ y))) if y.size else 0.0
    scale = max(aty, 1e-12)
    best_ratio, best_gap = float(ratio_grid[0]), float("inf")
    for ratio in ratio_grid:
        cfg = SolveConfig(k=int(k), lam=float(ratio) * scale, **kwargs)  # type: ignore[arg-type]
        try:
            res = solve_fn(A, y, int(k), cfg)  # type: ignore[call-arg]
        except Exception:
            continue
        r2 = float(np.sum((y - A @ res.x_hat) ** 2))  # type: ignore[attr-defined]
        gap = abs(r2 - target)
        if gap < best_gap:
            best_ratio, best_gap = float(ratio), float(gap)
    return best_ratio, best_gap


def search_all_baselines(
    A: np.ndarray, y: np.ndarray, k: int, ratio_grid: tuple[float, ...]
) -> dict[str, dict[str, float]]:
    """对每个基线做正则强度搜索，返回 ``{name: {"ratio", "gap"}}``。

    **公平性要求**：所有可调 lam 的基线（fista / amp / sk_lasso / sk_lars）都用
    同一网格、同一评分口径；不可调的（omp / cosamp / iht / sk_omp）标记
    ``ratio=-1``。门槛比较必须用本函数产出的基线，否则等于"基线调参不足"。
    """
    out: dict[str, dict[str, float]] = {}
    tunable = {"fista", "amp", "sk_lasso", "sk_lars"}
    for name in BASELINES:
        fn = SOLVERS.get(name)
        if fn is None:
            continue
        if name in tunable:
            ratio, gap = grid_search_ratio(fn, A, y, k, ratio_grid)  # type: ignore[arg-type]
            out[name] = {"ratio": ratio, "gap": gap}
        else:
            cfg = SolveConfig(k=int(k))
            try:
                fn(A, y, int(k), cfg)  # type: ignore[call-arg]
                out[name] = {"ratio": -1.0, "gap": float("nan")}
            except Exception:
                continue
    return out
