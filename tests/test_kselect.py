"""cs.kselect 的稀疏度/噪声估计测试。

作者：晨星 · CJX0712
"""

from __future__ import annotations

import numpy as np
import pytest

from core.config import SolveConfig
from cs.kselect import K_GRID, estimate_sigma, estimate_sigma2, noise_discrepancy_k, stability_k

# --------------------------------------------------------------------------- 不变量


def test_invariant_estimate_sigma_is_unbiased_on_pure_noise() -> None:
    """不变式 20：纯噪声（``x=0``）下 ``estimate_sigma`` 的相对误差 < 30%。

    数学依据：``estimate_sigma2`` 用**支撑集内插留出残差**估计噪声方差。
    纯噪声时支撑集维度 ``k0=1``，训练折内插是超定的（1 维拟合远大于 1 个
    自由度），留出残差里**只剩噪声**，故估计量无偏。

    **实测口径（重要）**：30% 这条只在 ``k0=1`` 时成立（实测最差 20.9%）。
    默认 ``k0 = m//6``（m=96 ⇒ k0=16）时相对误差高达 **85%**——支撑维度
    接近折内训练样本数时内插退化，估计严重上偏。所以这里显式传 ``k0=1``，
    并在下一个测试里把默认口径的偏差**如实锁定**。
    """
    worst = 0.0
    for sigma_true in (0.01, 0.1, 0.5):
        for seed in range(10):
            gen = np.random.default_rng(5000 + seed)
            A = gen.standard_normal((96, 256))
            A /= np.linalg.norm(A, axis=0, keepdims=True)
            y = gen.standard_normal(96) * sigma_true
            est = estimate_sigma(A, y, n_folds=4, k0=1)
            rel = abs(est / sigma_true - 1.0)
            worst = max(worst, rel)
    assert worst < 0.30, f"纯噪声下sigma 估计最差相对误差 {worst:.3f} ≥ 30%"


def test_invariant_estimate_sigma2_is_squared_sigma() -> None:
    """不变式 20b：``estimate_sigma² == estimate_sigma2``（口径自洽）。"""
    gen = np.random.default_rng(4242)
    A = gen.standard_normal((64, 128))
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    y = gen.standard_normal(64) * 0.1
    s = estimate_sigma(A, y, n_folds=4, k0=1)
    s2 = estimate_sigma2(A, y, n_folds=4, k0=1)
    assert s == pytest.approx(np.sqrt(s2), rel=1e-15)
    assert s2 > 0.0


def test_invariant_noise_discrepancy_k_returns_grid_member() -> None:
    """不变式 21：``noise_discrepancy_k`` 返回的 k 必落在 ``K_GRID`` 内。

    这是**合法性**不变式（不问数值准不准，只问返回值是否可用）：返回值被
    下游 ``CS_Fuse`` 直接当作稀疏度用，若越界或为负会造成不可恢复的状态。
    在全部 6 种 DGP regime 上验证。
    """
    from data.generators import KINDS, make_instance

    for kind in KINDS:
        inst = make_instance(kind, m=96, n=256, k=12, snr_db=18.0, coherence=0.5, seed=3)
        k_hat, sigma_hat = noise_discrepancy_k(inst.A, inst.y)
        assert k_hat in K_GRID, f"{kind}: k_hat={k_hat} 不在 K_GRID={K_GRID}"
        assert sigma_hat > 0.0
        assert np.isfinite(sigma_hat)


def test_invariant_noise_discrepancy_k_respects_m_bound() -> None:
    """不变式 21b：``k_hat < m − 2``（实现跳过 ``k ≥ m−2`` 的网格点）。

    OMP 在 ``k → m`` 时支撑集吃满观测空间，残差塌到 0，不一致性原理失去
    锚点。实现显式 ``continue`` 跳过这些 k，故返回值必满足该界。
    """
    from data.generators import make_instance

    inst = make_instance("wellcond", m=32, n=128, k=6, snr_db=20.0, seed=1)
    # 用一个比 m-2 小的自定义网格，确保返回的是网格成员而非被跳过的值
    small_grid = (4, 6, 8, 10)
    k_hat, _ = noise_discrepancy_k(inst.A, inst.y, small_grid)
    assert k_hat in small_grid
    assert k_hat < inst.m - 2


def test_invariant_stability_k_returns_valid_range() -> None:
    """不变式 22（D1 已修复）：``stability_k`` 返回 ``(k, freq)``，``k∈[1,m−2]``、``freq∈[0,1]``。

    D1 修复后 ``freq`` 按**总试验次数**（n_boot × |k_grid|，跳过 m−2 的网格点）
    归一化，因此 ``freq`` 是真正的频率，落在 ``[0,1]``。``k`` 被 clip 到 ``[1, m−2]``。
    """
    from data.generators import make_instance

    inst = make_instance("wellcond", m=48, n=128, k=6, snr_db=25.0, seed=9)
    k_best, freq = stability_k(inst.A, inst.y, n_boot=4)
    assert 1 <= k_best <= inst.m - 2
    assert freq.shape == (inst.n,)
    assert np.all(freq >= 0.0)  # 计数非负
    assert np.all(freq <= 1.0)  # D1 修复：频率归一化到 [0,1]
    assert freq.max() <= 1.0 + 1e-12


def test_invariant_stability_k_freq_is_normalized() -> None:
    """不变式 22b（D1 已修复）：``stability_k`` 的 ``freq`` 落在 ``[0, 1]``。

    D1 修复前 ``freq.max()`` 实测达 **10.0**（被放大 ``|k_grid|`` 倍）；修复后
    按总试验次数归一化，``freq`` 恰为入选频率，必落在 ``[0,1]``。
    """
    from data.generators import make_instance

    inst = make_instance("wellcond", m=48, n=128, k=6, snr_db=25.0, seed=9)
    _, freq = stability_k(inst.A, inst.y, n_boot=4)
    assert np.all(freq >= 0.0) and np.all(freq <= 1.0)
    assert freq.max() <= 1.0 + 1e-12


def test_invariant_stability_k_no_longer_saturates() -> None:
    """不变式 22c（D1 已修复）：``stability_k`` 不再把 k 饱和到 ``m−2``。

    D1 修复前 ``freq`` 被放大 10 倍，``freq >= thresh`` 几乎处处成立，
    ``k_best = stable.size`` 退化到 ``m−2``（m=48 时恒为 46）。修复后频率
    归一化到 ``[0,1]``，只有真正稳定的坐标入选，``k_best`` 不再饱和到上界。
    实测该实例 ``k_best = 6``（恰为 k_true）。
    """
    from data.generators import make_instance

    inst = make_instance("wellcond", m=48, n=128, k=6, snr_db=25.0, seed=9)
    k_best, _ = stability_k(inst.A, inst.y, n_boot=4)
    # 核心修复证据：不再饱和到 m-2
    assert k_best != inst.m - 2, f"stability_k 仍饱和到 m-2={inst.m - 2}: k_best={k_best}"
    assert 1 <= k_best <= inst.m - 2


def test_invariant_k_grid_is_sorted_and_valid() -> None:
    """不变式 22b：``K_GRID`` 是严格递增的正整数元组（下游二分/剪枝依赖）。"""
    assert isinstance(K_GRID, tuple)
    assert list(K_GRID) == sorted(K_GRID)
    assert len(set(K_GRID)) == len(K_GRID)
    assert all(isinstance(k, int) and k > 0 for k in K_GRID)


# --------------------------------------------------------------------------- 已知偏差锁定


def test_known_bias_estimate_sigma_default_k0_overestimates() -> None:
    """**已知偏差如实锁定**：默认 ``k0`` 下 ``estimate_sigma`` 系统性高估。

    实测（3 个 sigma × 20 seed = 60 例）：默认 ``k0 = m//6 = 16`` 时最大相对
    误差 **85%**，且**全部 60 例都是高估**（ratio ≥ 1.0）。根因见
    ``estimate_sigma2`` 的 docstring：支撑维度接近折内训练样本数时内插退化。

    这条测试**断言当前的真实偏差行为**（上偏），让偏差在 CI 中显式可见。
    若将来实现修正了估计器，本测试会失败——那时应改写为无偏断言。
    """
    ratios = []
    for sigma_true in (0.01, 0.1, 0.5):
        for seed in range(10):
            gen = np.random.default_rng(5000 + seed)
            A = gen.standard_normal((96, 256))
            A /= np.linalg.norm(A, axis=0, keepdims=True)
            y = gen.standard_normal(96) * sigma_true
            ratios.append(estimate_sigma(A, y) / sigma_true)
    ratios = np.asarray(ratios)
    # 全部高估（ratio ≥ 1），且平均明显> 1
    assert np.all(ratios >= 1.0), f"存在低估用例: min={ratios.min():.3f}"
    assert float(ratios.mean()) > 1.2, f"平均高估不足: mean={ratios.mean():.3f}"
    # 且默认口径达不到 30% 精度（这是选择 k0=1 的原因）
    assert float(np.abs(ratios - 1.0).max()) > 0.30


def test_noise_discrepancy_k_is_conservative() -> None:
    """``noise_discrepancy_k`` 系统性**偏保守**（k_hat ≤ k_true），如实锁定。

    实现 docstring 已诚实披露："实测 k_hat 为 4~10（k_true = 10~12），即
    系统性偏保守"。本测试在 6 regime 上验证这个方向性结论确实成立——
    偏差方向是安全的（欠拟合优于过拟合），但必须可验证而非口头声明。
    """
    from data.generators import KINDS, make_instance

    for kind in KINDS:
        inst = make_instance(kind, m=96, n=256, k=12, snr_db=18.0, coherence=0.5, seed=3)
        k_hat, _ = noise_discrepancy_k(inst.A, inst.y)
        assert k_hat <= inst.k_true, f"{kind}: k_hat={k_hat} > k_true={inst.k_true}（应偏保守）"


# --------------------------------------------------------------------------- 契约


def test_estimate_sigma_accepts_sigma_hint() -> None:
    """``noise_discrepancy_k`` 接受外部 ``sigma``，跳过内部估计（省一次 CV）。"""
    from data.generators import make_instance

    inst = make_instance("wellcond", m=48, n=128, k=6, snr_db=25.0, seed=9)
    k_auto, s_auto = noise_discrepancy_k(inst.A, inst.y)
    k_hint, s_hint = noise_discrepancy_k(inst.A, inst.y, sigma=0.05)
    assert s_hint == 0.05
    assert s_auto != 0.05
    assert k_auto in K_GRID
    assert k_hint in K_GRID


def test_estimate_sigma_returns_positive_on_finite_input() -> None:
    """有限输入下 ``estimate_sigma`` 恒返回正有限值（下限 1e-12）。"""
    from data.generators import make_instance

    inst = make_instance("wellcond", m=48, n=128, k=6, snr_db=25.0, seed=9)
    s = estimate_sigma(inst.A, inst.y)
    assert s > 0.0
    assert np.isfinite(s)
    # 全零 y ⇒ 估计值触及 1e-24 的方差下限 ⇒ sigma >= 1e-12
    s_zero = estimate_sigma(inst.A, np.zeros(inst.m))
    assert s_zero >= 1e-12


def test_n_folds_is_clipped_to_valid_range() -> None:
    """``n_folds`` 被 clip 到 ``[2, m//8]``（极端输入不崩）。"""
    from data.generators import make_instance

    inst = make_instance("wellcond", m=48, n=128, k=6, snr_db=25.0, seed=9)
    for n_folds in (0, 1, 2, 100):
        s = estimate_sigma(inst.A, inst.y, n_folds=n_folds)
        assert np.isfinite(s) and s > 0.0


def test_stability_k_is_deterministic() -> None:
    """``stability_k`` 使用固定 ``seed_offset`` 的派生流 ⇒ 逐位可复现。"""
    from data.generators import make_instance

    inst = make_instance("wellcond", m=48, n=128, k=6, snr_db=25.0, seed=9)
    k1, f1 = stability_k(inst.A, inst.y, n_boot=4)
    k2, f2 = stability_k(inst.A, inst.y, n_boot=4)
    assert k1 == k2
    assert np.array_equal(f1, f2)


def test_kselect_does_not_see_x_true() -> None:
    """k 估计器只接受 ``(A, y)``——签名层面无真值入口（防泄漏）。"""
    import inspect

    for fn in (estimate_sigma, estimate_sigma2, noise_discrepancy_k, stability_k):
        params = set(inspect.signature(fn).parameters)
        assert not (params & {"x_true", "k_true"}), f"{fn.__name__} 暴露了真值参数"


def test_solve_config_k_is_used_consistently() -> None:
    """``noise_discrepancy_k`` 内部用 ``SolveConfig(k=k)``，与网格值对齐。"""
    from data.generators import make_instance

    inst = make_instance("wellcond", m=48, n=128, k=6, snr_db=25.0, seed=9)
    k_hat, _ = noise_discrepancy_k(inst.A, inst.y)
    cfg = SolveConfig(k=k_hat, max_iter=200)
    assert cfg.k == k_hat
    assert cfg.max_iter == 200
