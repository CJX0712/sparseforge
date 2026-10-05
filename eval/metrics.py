"""eval.metrics — 恢复质量指标与跨 seed 聚合。

统一语义（与 core.types 一致）：
- ``nmse`` 越小越好
- ``support_f1`` 越大越好
- ``success`` = 支撑集完全恢复（f1 >= 1.0 - tol）

作者：晨星 · CJX0712
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from core.types import SUPPORT_EPS

__all__ = [
    "aggregate",
    "beats",
    "mean_std",
    "nmse",
    "success_of",
    "support_f1",
]

_EPS = 1e-10


def nmse(x_hat: np.ndarray, x_true: np.ndarray) -> float:
    d = float(np.sum(np.asarray(x_true, dtype=np.float64) ** 2))
    if d <= 0:
        return 0.0
    return float(np.sum((np.asarray(x_hat, dtype=np.float64) - x_true) ** 2) / d)


def support_f1(x_hat: np.ndarray, x_true: np.ndarray) -> float:
    pred = np.flatnonzero(np.abs(np.asarray(x_hat)) > SUPPORT_EPS)
    true = np.flatnonzero(np.abs(np.asarray(x_true)) > SUPPORT_EPS)
    if pred.size == 0 and true.size == 0:
        return 1.0
    if pred.size == 0 or true.size == 0:
        return 0.0
    inter = int(np.intersect1d(pred, true).size)
    if inter == 0:
        return 0.0
    prec = inter / pred.size
    rec = inter / true.size
    return float(2 * prec * rec / (prec + rec))


def success_of(f1: float, tol: float = 1e-9) -> bool:
    """支撑集完全恢复即判成功（跨噪声 regime 可用，不依赖 NMSE 绝对值）。"""
    return bool(f1 >= 1.0 - tol)


def mean_std(values: list[float] | np.ndarray) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def beats(
    a_mean: float, a_std: float, b_mean: float, b_std: float, sigma_factor: float = 0.5
) -> bool:
    """显著性判据：``a_mean < b_mean`` 且 ``b_mean - a_mean > sigma_factor*(a_std+b_std)``。

    比任务书 §4「均值差需大于两组 std 之和的 1/2」严格化：std 不可为 0 时
    退化为纯均值比较（由调用方保证至少 3 seeds ⇒ std > 0）。
    """
    if a_mean < b_mean:
        denom = a_std + b_std
        if denom <= 0.0:
            return True
        return bool((b_mean - a_mean) > sigma_factor * denom)
    return False


def aggregate(rows: list[dict]) -> dict[str, dict[str, object]]:
    """按 ``(instance, method)`` 聚合成 mean±std。

    输入为 ``BenchmarkRow.to_dict()`` 列表；返回
    ``{method: {"nmse_mean", "nmse_std", "f1_mean", "f1_std",
    "success_rate", "n", "per_instance": {...}}}``。
    """
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        buckets[(r["instance"], r["method"])].append(r)
    by_method: dict[str, list[dict]] = defaultdict(list)
    for (_inst, method), rs in buckets.items():
        by_method[method].extend(rs)
    out: dict[str, dict[str, object]] = {}
    for method, rs in sorted(by_method.items()):
        nm, ns = mean_std([r["nmse"] for r in rs])
        fm, fs = mean_std([r["support_f1"] for r in rs])
        per_inst: dict[str, dict[str, float]] = {}
        for (inst, meth), sub in buckets.items():
            if meth != method:
                continue
            a, b = mean_std([r["nmse"] for r in sub])
            c, d = mean_std([r["support_f1"] for r in sub])
            per_inst[inst] = {
                "nmse_mean": a,
                "nmse_std": b,
                "f1_mean": c,
                "f1_std": d,
                "n": len(sub),
            }
        out[method] = {
            "nmse_mean": nm,
            "nmse_std": ns,
            "f1_mean": fm,
            "f1_std": fs,
            "success_rate": float(np.mean([success_of(r["support_f1"]) for r in rs])),
            "n": len(rs),
            "per_instance": per_inst,
        }
    return out
