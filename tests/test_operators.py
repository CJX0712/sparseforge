"""cs.operators 的数学不变量测试。

本文件锁定的是**数值线性代数层面的可证明性质**（正交性、幂等性、单调性、
尺度不变性），而非"函数返回了一个数"。每个不变量一条独立测试。

作者：晨星 · CJX0712
"""

from __future__ import annotations

import numpy as np
import pytest

from core.errors import NumericsError
from core.seed import set_all
from cs.operators import (
    check_finite,
    coherence,
    debias_refit,
    oracle_nmse,
    soft_threshold,
    spectral_norm,
)

# --------------------------------------------------------------------------- 不变量


def test_invariant_orthogonal_matrix_is_orthogonal(rng: np.random.Generator) -> None:
    """不变式 1：均匀随机正交矩阵满足 ``QᵀQ = I``。

    QR 分解是构造正交矩阵的标准途径。``‖QᵀQ − I‖∞< 1e-12`` 是正交性的
    数值判据——若成立，则 ``Q⁻¹ = Qᵀ``，最小二乘的正规方程 ``AᵀAx = Aᵀb``
    在该子空间上有唯一解，这正是 OMP 支撑集重拟合的理论依据。
    """
    gen = np.random.default_rng(7)
    q, r = np.linalg.qr(gen.standard_normal((64, 64)))
    # QR 的 R 对角可能为负，统一符号使 Q 成为唯一确定的正交矩阵
    q = q * np.sign(np.diag(r))[None, :]
    assert np.abs(q.T @ q - np.eye(64)).max() < 1e-12


def test_invariant_soft_threshold_is_not_idempotent() -> None:
    """不变式 2：**软阈值算子不是幂等的**——这是一条被广泛误解的性质。

    ``soft(v, λ) = sign(v)·max(|v|−λ, 0)``。对任意 ``|v| > λ`` 有
    ``|soft(v,λ)| = |v| − λ > 0``，再次收缩会**再减一次** λ，于是精确地有

    ``soft(soft(v, λ), λ) == soft(v, 2λ)``   （对 ``λ ≥ 0`` 恒成立）

    因此本实现**不**满足 ``‖soft(soft(v,λ),λ) − soft(v,λ)‖∞ < 1e-15``——那条
    断言在数学上是错的（真正幂等的是**硬**阈值算子）。本测试锁定两件真事：
    （a）二次收缩等价于一次 ``2λ`` 收缩；（b）幂等性确实不成立。
    """
    v = np.array([-5.0, -1.3, -0.2, 0.0, 0.2, 1.3, 5.0])
    lam = 1.0
    once = soft_threshold(v, lam)
    twice = soft_threshold(once, lam)
    # (a) 二次收缩严格等价于单次 2λ 收缩——这是精确恒等式，不是近似
    assert np.array_equal(twice, soft_threshold(v, 2.0 * lam))
    # (b) 幂等性不成立：|v| > λ 的分量两次结果确实不同
    assert np.abs(twice - once).max() > 0.0
    # λ=0 是唯一幂等的非负阈值
    assert np.array_equal(soft_threshold(v, 0.0), v)


def test_invariant_soft_threshold_is_nonexpansive() -> None:
    """不变式 3：软阈值是**非扩张**算子，``|soft(v,λ)| ≤ |v|`` 逐分量成立。

    这是软阈值在凸优化中的核心收敛性依据：它不会放大任何分量，故 FISTA
    的近端梯度步在 ℓ1 意义下单调。实现为 ``np.sign(v)·max(|v|−λ, 0)``，
    当 ``λ ≥ 0`` 时逐分量不超过 ``|v|``。
    """
    gen = np.random.default_rng(11)
    v = gen.standard_normal(500) * 4.0
    for lam in (0.0, 0.1, 1.0, 3.7):
        assert np.all(np.abs(soft_threshold(v, lam)) <= np.abs(v))
    # λ=0 时是恒等映射
    assert np.array_equal(soft_threshold(v, 0.0), v)
    # |v| ≤ λ 的分量被完全置零
    w = np.array([-0.5, -0.5, 0.5, 0.5])
    assert np.array_equal(soft_threshold(w, 0.5), np.zeros(4))


def test_invariant_coherence_lies_in_unit_interval() -> None:
    """不变式 4：相干度必落在 ``[0, 1]``。

    相干度定义为最大列间内积绝对值 ``max_{i≠j} |⟨a_i, a_j⟩|``，由Cauchy–Schwarz
    不等式其上界为 1（两列共线时取到），下界为 0（列正交时取到）。实现里先
    按列归一化再取 Gram 矩阵最大值，故该界必须自动满足——这条测试锁死"实现
    确实是按定义算的"（而不是漏了归一化导致 >1）。
    """
    gen = np.random.default_rng(3)
    cases = [
        gen.standard_normal((20, 60)),  # 随机矩阵
        np.eye(6),  # 完全正交 ⇒ 0
        np.hstack([np.ones((5, 1))] * 4),  # 完全共线 ⇒ 1
        np.zeros((7, 9)),  # 全零（实现把零范数列当作单位列）
        gen.standard_normal((8, 1)),  # 单列 ⇒ 无列对 ⇒ 0
    ]
    for A in cases:
        mu = coherence(A)
        assert 0.0 <= mu <= 1.0, f"coherence 越界: {mu}"


def test_invariant_coherence_is_scale_invariant() -> None:
    """不变式 5：``coherence(c ⊙ A) == coherence(A)``——相干度对列缩放不变。

    数学依据：``⟨c_i a_i, c_j a_j⟩ = c_i c_j ⟨a_i, a_j⟩``，归一化后列缩放因子
    被约掉。实现确实先除以列范数，因此**尺度不变**（这条成立，写为测试）。
    同时验证该不变性对随机正缩放与乱序缩放都成立。
    """
    gen = np.random.default_rng(5)
    A = gen.standard_normal((20, 50))
    base = coherence(A)
    # 单列缩放
    col_scale = np.arange(1, 51, dtype=np.float64) * 1.7
    assert coherence(A * col_scale) == pytest.approx(base, abs=1e-12)
    # 统一正缩放
    assert coherence(A * 3.5) == pytest.approx(base, abs=1e-12)
    # 随机正缩放
    rand_scale = gen.uniform(0.1, 10.0, size=50)
    assert coherence(A * rand_scale) == pytest.approx(base, abs=1e-12)


def test_invariant_spectral_norm_matches_dense_svd() -> None:
    """不变式 6：幂迭代 ``spectral_norm`` 与 ``np.linalg.norm(A, 2)`` 收敛到同一值。

    谱范数 ``‖A‖₂ = σ_max(A)``。实现用**固定初始向量**的幂迭代（因此是确定
    性的，重复调用逐位相同）。

    **实测说明（重要）**：默认 ``n_iter=30`` 的精度**强烈依赖谱间隙**。在随机
    高斯小矩阵上实测相对误差从 6e-16 到 **5.5e-2** 不等——最差的
    ``(10, 25)`` 样本谱间隙仅 ``(σ₁−σ₂)/σ₁ ≈ 0.04``，30 步幂迭代远未收敛。
    因此本测试断言的是**可证的两件事**：

    1. 对**有谱间隙**的矩阵族（正交列 / 秩一 / 对角），30 步即精确到机器精度
    2. 迭代数增加时误差收敛到 0（``n_iter=300`` 时所有形状相对误差 < 1e-7）

    不断言"默认 30 步在任意随机矩阵上都精确"——那在数学上是假的。
    """
    # (1) 正交列矩阵：σ_max = 1，幂迭代一步到位
    for seed in range(4):
        gen = np.random.default_rng(100 + seed)
        q, _ = np.linalg.qr(gen.standard_normal((20, 20)))
        B = q[:12].T  # (20, 12)，列正交且单位范数
        assert spectral_norm(B) == pytest.approx(1.0, rel=1e-12)

    # (2) 迭代数 → 误差单调收敛到机器精度（覆盖含小谱间隙的形状）
    for shape in [(8, 20), (10, 25), (12, 30), (9, 18)]:
        B = np.random.default_rng(7).standard_normal(shape)
        exact = float(np.linalg.norm(B, 2))
        assert spectral_norm(B, n_iter=300) == pytest.approx(exact, rel=1e-7)
        # 默认 30 步至少给出一个**下界意义**的粗略估计：不差到离谱
        assert 0.5 * exact <= spectral_norm(B) <= 2.0 * exact

    # 固定初值 ⇒ 确定性：同参数重复调用逐位相同
    B = np.random.default_rng(1).standard_normal((10, 25))
    assert spectral_norm(B) == spectral_norm(B)


def test_invariant_spectral_norm_edge_cases() -> None:
    """不变式 6b：谱范数在退化输入上的边界行为。

    - 全零矩阵 ⇒ σ_max = 0，实现须返回 0.0 而非 ``1e-12`` 下溢
    - 对角阵的谱范数是最大对角元绝对值（可精确验证）
    - 秩一矩阵 ``uvᵀ`` 的谱范数是 ``‖u‖·‖v‖``（可精确验证）
    """
    assert spectral_norm(np.zeros((4, 6))) == 0.0
    assert spectral_norm(np.diag([3.0, 1.0, 0.5])) == pytest.approx(3.0, rel=1e-6)
    u = np.arange(1.0, 9.0)
    v = np.arange(1.0, 21.0)
    rank1 = np.outer(u, v)
    exact = float(np.linalg.norm(u) * np.linalg.norm(v))
    assert spectral_norm(rank1) == pytest.approx(exact, rel=1e-6)


def test_invariant_debias_refit_never_increases_residual() -> None:
    """不变式 7：``debias_refit`` 后的残差范数 ≤ 去偏前的残差范数。

    数学依据：``debias_refit`` 在给定支撑集上做**最小二乘**拟合，
    ``min_{c} ‖y − A_S c‖₂``。最小二乘解是全空间的全局最小值，故任何其他
    系数（特别是阈值类方法产生的收缩/放大解）的残差都不可能更小。

    测试构造一个"未去偏"的解：把真实系数整体乘 0.6（模拟阈值带来的幅度
    收缩偏差——这正是 debias 要修正的对象），然后断言 LS 重拟合后残差下降。
    """
    gen = np.random.default_rng(9)
    A = gen.standard_normal((50, 120))
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    support = np.array([1, 7, 23, 44, 60])
    coef = gen.standard_normal(5) * 2.0
    y = A[:, support] @ coef + 0.01 * gen.standard_normal(50)

    # 未去偏：幅度被收缩 0.6（阈值法的典型产物）
    x_shrunk = np.zeros(120)
    x_shrunk[support] = coef * 0.6
    res_before = float(np.linalg.norm(y - A @ x_shrunk))

    x_debiased = debias_refit(A, y, support)
    res_after = float(np.linalg.norm(y - A @ x_debiased))

    assert res_after <= res_before + 1e-12
    # 且确实严格下降（否则测试无意义）
    assert res_after < res_before
    # 去偏解就是该支撑集上的 LS 最优解：与 lstsq 直接求解逐位一致
    coef_ref, *_ = np.linalg.lstsq(A[:, support], y, rcond=None)
    assert np.allclose(x_debiased[support], coef_ref, atol=1e-10)
    # 支撑集之外严格为零
    mask = np.ones(120, dtype=bool)
    mask[support] = False
    assert np.all(x_debiased[mask] == 0.0)


def test_invariant_debias_refit_monotone_on_solution_family() -> None:
    """不变式 7b：对同一支撑集上的**任意**系数缩放，LS 重拟合都是最优的。

    不变式 7 只验证了一个点。这条把它推广到一族解：遍历 ``scale ∈ (0, 2]``，
    断言 ``debias_refit`` 的残差始终 ≤ 该族中每个成员的残差。这排除了
    "碰巧在某个点上成立"的可能。
    """
    gen = np.random.default_rng(23)
    A = gen.standard_normal((40, 90))
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    support = np.array([2, 15, 33, 61, 77])
    coef = gen.standard_normal(5) * 1.5
    y = A[:, support] @ coef + 0.05 * gen.standard_normal(40)
    ls_res = float(np.linalg.norm(y - A @ debias_refit(A, y, support)))
    for scale in np.linspace(0.05, 2.0, 25):
        x = np.zeros(90)
        x[support] = coef * scale
        assert float(np.linalg.norm(y - A @ x)) >= ls_res - 1e-12


def test_invariant_oracle_nmse_is_zero_iff_exact() -> None:
    """不变式 8：``oracle_nmse`` 精确恢复时为 0，且与 ``eval.metrics.nmse`` 定义一致。

    ``nmse = ‖x̂−x‖²/‖x‖²``：分子为 0 当且仅当 ``x̂ == x``。同时验证两个模块
    （``cs.operators.oracle_nmse`` 与 ``eval.metrics.nmse``）口径完全一致——
    它们是同一指标的两次实现，不允许漂移。
    """
    from eval.metrics import nmse as metrics_nmse

    gen = np.random.default_rng(31)
    x_true = gen.standard_normal(30)
    assert oracle_nmse(x_true, x_true) == 0.0
    assert metrics_nmse(x_true, x_true) == 0.0
    other = gen.standard_normal(30)
    assert oracle_nmse(other, x_true) == pytest.approx(metrics_nmse(other, x_true), rel=1e-15)
    # 分母为 0（x_true 全零）时约定返回 0.0 而非除零
    zeros = np.zeros(5)
    assert oracle_nmse(np.ones(5), zeros) == 0.0
    assert metrics_nmse(np.ones(5), zeros) == 0.0


# --------------------------------------------------------------------------- 契约/边界


def test_coherence_zero_and_identity_columns() -> None:
    """边界：正交矩阵相干度为 0，完全共线为 1，单列为 0。"""
    assert coherence(np.eye(6)) == pytest.approx(0.0, abs=1e-12)
    assert coherence(np.hstack([np.ones((5, 1))] * 4)) == pytest.approx(1.0, abs=1e-12)
    assert coherence(np.random.default_rng(2).standard_normal((8, 1))) == 0.0


def test_debias_refit_rejects_out_of_bounds_support() -> None:
    """支撑集越界必须抛 :class:`NumericsError`，而非静默截断或IndexError。"""
    A = np.random.default_rng(3).standard_normal((10, 20))
    y = np.random.default_rng(4).standard_normal(10)
    with pytest.raises(NumericsError, match="支撑集越界"):
        debias_refit(A, y, np.array([0, 99]))


def test_debias_refit_empty_support_returns_zeros() -> None:
    """空支撑集返回全零向量（长度 n），不抛错——这是合法的"什么都没选上"。"""
    A = np.random.default_rng(3).standard_normal((10, 20))
    y = np.random.default_rng(4).standard_normal(10)
    out = debias_refit(A, y, np.array([], dtype=int))
    assert out.shape == (20,)
    assert np.all(out == 0.0)


def test_check_finite_raises_on_non_finite() -> None:
    """``check_finite`` 对 NaN/Inf 均抛 :class:`NumericsError`。"""
    with pytest.raises(NumericsError, match="NaN/Inf"):
        check_finite(np.array([1.0, np.nan]))
    with pytest.raises(NumericsError, match="NaN/Inf"):
        check_finite(np.array([np.inf]))
    with pytest.raises(NumericsError, match="NaN/Inf"):
        check_finite(np.array([1.0]), np.array([np.nan]))
    # 有限输入静默通过
    check_finite(np.zeros(3), np.ones((2, 2)))


def test_operators_module_reexports_are_importable() -> None:
    """``__all__`` 中每个名字都真实可导入（防止改名后静默失效）。"""
    import cs.operators as ops

    for name in ops.__all__:
        assert hasattr(ops, name), f"cs.operators 声明导出 {name} 但属性不存在"


def test_set_all_rejects_non_integer_seed() -> None:
    """:func:`core.seed.set_all` 只接受整数；bool 被显式拒绝（bool 是 int 子类）。"""
    from core.errors import ConfigError

    with pytest.raises(ConfigError, match="seed 必须是整数"):
        set_all("not-an-int")  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="seed 必须是整数"):
        set_all(True)  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match=r"\[0, 2\*\*63\)"):
        set_all(-1)
    # 合法 seed 被接受并原样返回
    assert set_all(12345) == 12345
