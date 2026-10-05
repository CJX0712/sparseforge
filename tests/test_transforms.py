"""preprocess.transforms 的白化/逆变换测试。

作者：晨星 · CJX0712
"""

from __future__ import annotations

import numpy as np
import pytest

from data.generators import make_instance
from preprocess.transforms import TransformPipeline, gram_whitener


def _inst() -> object:
    return make_instance("wellcond", m=48, n=128, k=6, snr_db=25.0, seed=9)


def test_gram_whitener_works_underdetermined() -> None:
    """不变式 47（P0 已修复）：``m < n`` 下 ``gram_whitener`` 不复崩且白化有效。

    **修复前**：``G = A.T @ A`` 是 ``n×n``，但收缩锚点用了 ``np.eye(m)``
    （``m < n``），广播失配直接抛 ``ValueError``，使 ``TransformPipeline``
    在压缩感知的 ``m<n`` 设定下完全不可用。修复后锚点改为 ``np.eye(n)``，
    且白化矩阵取 ``W = L^{-T}``（``G_shrunk = L L^T``），使
    ``(A W)^T (A W) = I``。
    """
    inst = _inst()
    w = gram_whitener(inst.A, shrink=0.02)
    assert w.W.shape == (inst.n, inst.n)
    assert np.all(np.isfinite(w.W))
    assert np.all(np.isfinite(w.Winv))
    # 白化后算子接近正交：条件数从 ~3.9 降到 ~1.06
    Aw = inst.A @ w.W
    assert np.linalg.cond(Aw) < 1.5, f"白化后条件数仍高: {np.linalg.cond(Aw):.3f}"
    assert np.linalg.cond(Aw) < np.linalg.cond(inst.A)


def test_gram_whitener_inverse_recovers_signal() -> None:
    """不变式 47b：白化空间解经逆变换逐位还原原始信号。

    给定真值 ``x``，其白化空间表示为 ``x_w = W^{-1} x``；流水线
    ``inverse_transform`` 用 ``out @ Winv``（= ``Winv^T @ out``）回放，
    应得 ``x``（要求 ``Winv^T = W``，即 ``Winv = L^{-1}``）。
    """
    inst = _inst()
    w = gram_whitener(inst.A, shrink=0.02)
    rng = np.random.default_rng(7)
    x = rng.standard_normal(inst.n)
    xw = np.linalg.inv(w.W) @ x  # 原始 -> 白化空间
    x_rec = xw @ w.Winv  # 流水线逆变换（out @ Winv）
    assert np.allclose(x_rec, x, atol=1e-10)


def test_pipeline_transform_inverse_roundtrip() -> None:
    """``TransformPipeline`` 在 ``m<n`` 下不崩，且逆变换与白化逆一致。"""
    inst = _inst()
    pipe = TransformPipeline(center=False, whiten=True, shrink=0.02)
    A_t, _y_t = pipe.fit_transform(inst.A, inst.y)
    assert A_t.shape == inst.A.shape
    assert np.all(np.isfinite(A_t))
    rng = np.random.default_rng(11)
    xw = rng.standard_normal(inst.n)
    x_rec = pipe.inverse_transform(xw)
    w = gram_whitener(inst.A, shrink=0.02)
    assert np.allclose(x_rec, xw @ w.Winv, atol=1e-10)


def test_residual_center_removes_mean() -> None:
    """``residual_center`` 把 y 去均值，且返回 ``mu`` 等于原 y 均值。"""
    from preprocess.transforms import residual_center

    inst = _inst()
    _A_c, y_c, mu = residual_center(inst.A, inst.y)
    assert abs(float(np.mean(y_c))) < 1e-9
    assert mu == pytest.approx(float(np.mean(inst.y)))
    # y_c + mu 还原原 y 的均值
    assert float(np.mean(y_c + mu)) == pytest.approx(float(np.mean(inst.y)))
