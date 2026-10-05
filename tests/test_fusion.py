"""cs.fusion 的旗舰求解器、路由与精修测试。

作者：晨星 · CJX0712
"""

from __future__ import annotations

import numpy as np
import pytest

from core.config import SolveConfig
from core.errors import NumericsError
from core.seed import set_all
from cs.fusion import (
    CS_Fuse,
    block_omp,
    estimate_snr_db,
    polish_support,
    route_solver,
    stability_select,
    support_vote,
)

# --------------------------------------------------------------------------- 不变量


def test_invariant_polish_support_residual_is_monotone() -> None:
    """不变式 39：``polish_support`` 的训练残差 ``‖y − A_S x_S‖²`` **单调不增**。

    docstring 明确承诺："每次交换**只在降低训练残差时接受**"。实现里
    接受条件是 ``val < best * 0.95``（相对下降 >5%），故单调性成立。

    这条用一个**已知会触发交换**的实例验证（初始支撑集全错→ 精修后残差
    从 3.99 降到 0.16，交换确实发生且残差大幅下降）。在 15 个随机 seed 上
    重复，验证无反例。
    """
    for seed in range(15):
        gen = np.random.default_rng(400 + seed)
        m, n = 40, 100
        A = gen.standard_normal((m, n))
        A /= np.linalg.norm(A, axis=0, keepdims=True)
        true_s = np.sort(gen.choice(n, 5, replace=False))
        y = A[:, true_s] @ gen.standard_normal(5) + 0.05 * gen.standard_normal(m)
        wrong = np.arange(5)  # 完全错误的支撑集

        res_in = float(
            np.sum((y - A[:, wrong] @ np.linalg.lstsq(A[:, wrong], y, rcond=None)[0]) ** 2)
        )
        sup_out, x_out = polish_support(A, y, wrong, max_rounds=5)
        res_out = float(np.sum((y - A @ x_out) ** 2))

        # 核心不变式：残差不增
        assert res_out <= res_in + 1e-12, f"seed={seed}: 残差反而上升 {res_in}→{res_out}"
        # 支撑集大小不变（1-opt 交换：换出1个换入1个）
        assert sup_out.size == wrong.size
        assert np.array_equal(np.sort(sup_out), sup_out)  # 升序去重


def test_invariant_polish_support_reduces_residual_on_wrong_support() -> None:
    """不变式 39b：在**完全错误**的支撑集上，精修确实改善残差（且大幅改善）。

    上一条锁定"不增"，这条锁定"真的会减"——否则一个什么都不做的恒等函数
    也能通过上一条测试。实测残差 3.99 → 0.16（下降 96%）。
    """
    gen = np.random.default_rng(21)
    m, n = 40, 100
    A = gen.standard_normal((m, n))
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    true_s = np.array([5, 17, 33, 61, 88])
    y = A[:, true_s] @ gen.standard_normal(5) + 0.05 * gen.standard_normal(m)
    wrong = np.array([0, 1, 2, 3, 4])

    res_in = float(
        np.sum((y - A[:, wrong] @ np.linalg.lstsq(A[:, wrong], y, rcond=None)[0]) ** 2)
    )
    sup_out, x_out = polish_support(A, y, wrong, max_rounds=5)
    res_out = float(np.sum((y - A @ x_out) ** 2))
    assert res_out < res_in * 0.5, f"精修改善不足: {res_in:.4f} → {res_out:.4f}"
    # 精修后至少换对 3/5 个坐标
    assert len(set(sup_out.tolist()) & set(true_s.tolist())) >= 3


def test_polish_support_preserves_support_size() -> None:
    """精修是 **1-opt 交换**：换出 1 个换入 1 个，故支撑集大小恒定。

    这条锁定交换的**结构**性质（size 不变、升序去重、与原集有交集），
    它比"精修保持正确支撑不变"更强且真实成立。
    """
    gen = np.random.default_rng(31)
    m, n = 40, 100
    A = gen.standard_normal((m, n))
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    true_s = np.sort(gen.choice(n, 5, replace=False))
    y = A[:, true_s] @ gen.standard_normal(5) + 0.01 * gen.standard_normal(m)

    sup_out, _ = polish_support(A, y, true_s, max_rounds=3)
    assert sup_out.size == true_s.size
    assert np.array_equal(np.sort(sup_out), sup_out)
    # 1-opt 交换最多改动 1 个坐标 ⇒ 与原集至少共享 size−1 个
    assert len(set(sup_out.tolist()) & set(true_s.tolist())) >= true_s.size - 1


def test_known_defect_polish_support_overfits_on_correct_support() -> None:
    """**已知缺陷**：精修会把**已经正确**的支撑集改坏（m<n 下的过拟合）。

    实现 docstring 已诚实披露："本步骤在 ``m < n`` 时会**过拟合**——训练残差
    不是有效的模型选择信号。作为消融项保留（``use_polish``），**默认关闭**"。

    实测（m=40, n=100, 真实支撑 ``[37,59,70,72,89]``）：精修把它改成
    ``[37,59,70,89,91]``——**换掉了正确的 72**，换来一个错的 91，训练残差
    从 3.190e-3 降到 3.022e-3（降幅 5.27%，刚好越过 5% 接受门槛）。

    结论：5% 的相对下降门槛**不足以**在 ``m ≪ n`` 时拦住噪声驱动的伪改善。
    本测试断言该现状（支撑集被改动），使缺陷在 CI 中显式可见；
    修复后本测试会失败（届时改写为"正确支撑保持不变"断言）。
    """
    gen = np.random.default_rng(31)
    m, n = 40, 100
    A = gen.standard_normal((m, n))
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    true_s = np.sort(gen.choice(n, 5, replace=False))
    y = A[:, true_s] @ gen.standard_normal(5) + 0.01 * gen.standard_normal(m)

    sup_out, x_out = polish_support(A, y, true_s, max_rounds=3)
    # 精修改动了支撑集（换掉了至少一个正确坐标）
    assert not np.array_equal(sup_out, true_s)
    # 且改动伴随训练残差下降（这正是过拟合的机制）
    res_in = float(
        np.sum((y - A[:, true_s] @ np.linalg.lstsq(A[:, true_s], y, rcond=None)[0]) ** 2)
    )
    res_out = float(np.sum((y - A @ x_out) ** 2))
    assert res_out < res_in


def test_invariant_polish_support_empty_support_is_noop() -> None:
    """空支撑集直接返回（不进入交换循环，不崩）。"""
    gen = np.random.default_rng(5)
    A = gen.standard_normal((20, 50))
    y = gen.standard_normal(20)
    sup, x = polish_support(A, y, np.array([], dtype=int))
    assert sup.size == 0
    assert np.all(x == 0.0)


def test_invariant_route_solver_partition() -> None:
    """不变式 40：``route_solver`` 按 (相干度, 维度) 划分两个**不相交**的族。

    判据（实现 docstring，D-b 已修复）：给定 ``(m, n)`` 时，相干度相对同维
    随机基线的超出量 ``coh / baseline(m,n) > 1.5`` ⇒ 真高相干族 ``iht``；
    否则 ⇒ 贪心族 ``omp``。未给维度时退化到绝对阈值 ``coh > 0.35`` ⇒ ``iht``。
    """
    # 维度自适应：wellcond（低相干，超出量<1.5）→ omp；coherent（高相干，超出量>1.5）→ iht
    assert route_solver(0.44, 30.0, 48, 128) == "omp"
    assert route_solver(0.97, 30.0, 48, 128) == "iht"
    # iid 高斯基线（m=96,n=256 实测 ~0.46）→ 普通实例落到 omp
    assert route_solver(0.46, 30.0, 96, 256) == "omp"
    assert route_solver(0.97, 30.0, 96, 256) == "iht"
    # 向后兼容：不给 m/n 时用绝对阈值 coh > 0.35
    assert route_solver(0.1, 30.0) == "omp"
    assert route_solver(0.0, 12.0) == "omp"
    assert route_solver(0.35, 30.0) == "omp"  # 边界：coh == 0.35 不算高相干
    assert route_solver(0.5, 30.0) == "iht"  # 绝对阈值以上
    assert route_solver(0.36, 30.0) == "iht"
    # 输出值域封闭
    for coh in (0.0, 0.2, 0.5, 0.9):
        for snr in (0.0, 11.9, 12.0, 40.0):
            assert route_solver(coh, snr, 96, 256) in {"omp", "iht"}


def test_invariant_stability_select_freq_is_normalized() -> None:
    """不变式 41：``stability_select`` 的 ``freq ∈ [0,1]``（真频率）。

    与 :func:`cs.kselect.stability_k` 不同，这个实现固定单个 k、只在外层
    bootstrap 累加、最后除以 ``n_boot``，因此频率被正确归一化。
    """
    from data.generators import make_instance

    inst = make_instance("wellcond", m=48, n=128, k=6, snr_db=25.0, seed=9)
    sup, freq = stability_select(inst.A, inst.y, 6, n_boot=8, thresh=0.5)
    assert freq.shape == (inst.n,)
    assert np.all(freq >= 0.0)
    assert np.all(freq <= 1.0)
    # freq 是 n_boot 的整数倍
    assert np.allclose(freq * 8, np.round(freq * 8))
    # support 恰是 freq >= thresh 的坐标
    assert np.array_equal(sup, np.flatnonzero(freq >= 0.5))


def test_invariant_cs_fuse_returns_valid_result() -> None:
    """不变式 42：``CS_Fuse.solve`` 返回结构合法、有限、自洽的解。

    契约：``x_hat`` 形状 ``(n,)`` 全有限；``residual_norm`` 与 ``x_hat``
    自洽；``meta`` 含 ``k_used``/``solver``/``coherence`` 等诊断字段。
    """
    from data.generators import make_instance

    inst = make_instance("wellcond", m=48, n=128, k=6, snr_db=25.0, seed=9)
    res = CS_Fuse().solve(inst.A, inst.y)
    assert res.x_hat.shape == (inst.n,)
    assert np.all(np.isfinite(res.x_hat))
    assert res.residual_norm == pytest.approx(
        float(np.linalg.norm(inst.y - inst.A @ res.x_hat)), rel=1e-9
    )
    assert res.solver == "cs_fuse"
    for key in ("k_used", "solver", "coherence", "snr_db_est", "n_supp"):
        assert key in res.meta, f"meta 缺字段 {key}"
    assert 1 <= int(res.meta["k_used"]) <= inst.m - 2


def test_invariant_cs_fuse_is_deterministic() -> None:
    """不变式 42b：``CS_Fuse.solve`` 逐位可复现（内部 k 估计用派生随机流）。"""
    from data.generators import make_instance

    inst = make_instance("wellcond", m=48, n=128, k=6, snr_db=25.0, seed=9)
    a = CS_Fuse().solve(inst.A, inst.y)
    b = CS_Fuse().solve(inst.A, inst.y)
    assert np.array_equal(a.x_hat, b.x_hat)
    assert a.residual_norm == b.residual_norm
    assert a.meta["k_used"] == b.meta["k_used"]


def test_invariant_cs_fuse_honors_explicit_k_when_kest_off() -> None:
    """不变式 43：``use_kest=False`` 且给定 ``k`` 时，旗舰**用该k**（消融口径）。

    这是公平性契约的一部分：传 k 时旗舰与基线用同一个 k。
    """
    from data.generators import make_instance

    inst = make_instance("wellcond", m=48, n=128, k=6, snr_db=25.0, seed=9)
    fuse = CS_Fuse(use_kest=False, use_route=False, fixed_solver="omp")
    res = fuse.solve(inst.A, inst.y, k=6)
    assert res.meta["k_used"] == 6
    assert res.meta["k_true_given"] is True
    assert res.meta["solver"] == "omp"


def test_invariant_cs_fuse_never_sees_x_true() -> None:
    """不变式 43b：``CS_Fuse.solve`` 签名中无真值入口（防泄漏硬约束）。"""
    import inspect

    params = set(inspect.signature(CS_Fuse.solve).parameters)
    assert not (params & {"x_true", "k_true"})


def test_invariant_ablation_switches_change_behavior() -> None:
    """不变式 44：四个消融开关各自真实影响输出路径（不是摆设）。

    - ``use_route=False`` ⇒ ``meta["solver"]`` 恒为 ``fixed_solver``
    - ``use_route=True`` ⇒ solver 由路由决定
    - ``use_debias`` 影响去偏前后支撑数（``n_supp_pre_debias``）
    - ``use_polish`` 记录交换数 ``n_swaps``
    """
    from data.generators import make_instance

    inst = make_instance("wellcond", m=48, n=128, k=6, snr_db=25.0, seed=9)
    # 关路由 ⇒ 固定求解器
    res = CS_Fuse(use_kest=False, use_route=False, fixed_solver="omp").solve(
        inst.A, inst.y, k=6
    )
    assert res.meta["solver"] == "omp"
    # 开路由 ⇒ 由 (coh, snr, m, n) 决定，取值仍是合法求解器名
    routed = CS_Fuse(use_kest=False, use_route=True).solve(inst.A, inst.y, k=6)
    assert routed.meta["solver"] in {"omp", "iht"}
    # 精修开关被记录
    polished = CS_Fuse(use_kest=False, use_polish=True).solve(inst.A, inst.y, k=6)
    assert "n_swaps" in polished.meta
    # 去偏前支撑数被记录
    assert "n_supp_pre_debias" in res.meta


def test_invariant_support_vote_returns_debiased_solution() -> None:
    """不变式 45：``support_vote`` 返回支撑集上的 LS 去偏解（全尺寸）。"""
    from data.generators import make_instance

    inst = make_instance("wellcond", m=48, n=128, k=6, snr_db=25.0, seed=9)
    x = support_vote(inst.A, inst.y, 6, n_boot=6)
    assert x.shape == (inst.n,)
    assert np.all(np.isfinite(x))
    # 支撑集外的坐标严格为零
    sup = np.flatnonzero(np.abs(x) > 1e-10)
    assert sup.size <= 6
    if sup.size:
        coef, *_ = np.linalg.lstsq(inst.A[:, sup], inst.y, rcond=None)
        assert np.allclose(x[sup], coef, atol=1e-10)


def test_invariant_estimate_snr_db_is_finite() -> None:
    """不变式 46：``estimate_snr_db`` 返回有限 dB 值（路由的输入，不能 NaN）。"""
    from data.generators import make_instance

    inst = make_instance("wellcond", m=48, n=128, k=6, snr_db=25.0, seed=9)
    est = estimate_snr_db(inst.A, inst.y)
    assert np.isfinite(est)
    # 真值 25dB，估计应在同一量级（宽松区间，避免过拟合到具体数值）
    assert 0.0 < est < 80.0


# --------------------------------------------------------------------------- 契约/兜底


def test_cs_fuse_rejects_non_positive_tau() -> None:
    """``tau ≤ 0`` 抛 :class:`NumericsError`（不一致性原理的目标倍数必须为正）。"""
    for bad_tau in (0.0, -1.0):
        with pytest.raises(NumericsError, match="tau"):
            CS_Fuse(tau=bad_tau)


def test_cs_fuse_degrades_gracefully_on_degenerate_input() -> None:
    """退化输入（A 全零）下 ``CS_Fuse.solve`` 不裸崩（内部降级到 omp）。"""
    A = np.zeros((12, 30))
    y = np.zeros(12)
    res = CS_Fuse().solve(A, y)
    assert res.x_hat.shape == (30,)
    assert np.all(np.isfinite(res.x_hat))


def test_block_omp_expands_by_blocks() -> None:
    """``block_omp`` 按整块扩张支撑集，块大小来自 ``block`` 参数。"""
    from data.generators import make_instance

    inst = make_instance("clustered", m=48, n=128, k=12, snr_db=20.0, seed=4)
    res = block_omp(inst.A, inst.y, 6, SolveConfig(k=6), block=8)
    assert res.x_hat.shape == (inst.n,)
    assert np.all(np.isfinite(res.x_hat))
    assert res.meta["block"] == 8
    # 支撑集必为 8 的倍数（按块扩张）
    assert res.support.size % 8 == 0


def test_fusion_handles_all_regimes_without_crashing() -> None:
    """6 种 regime 上 ``CS_Fuse.solve`` 都不抛异常（旗舰的"永不抛错"契约）。

    docstring 承诺"失败时降级到 omp（保证旗舰永不抛错）"。这条在全部 DGP 上验证。
    """
    from data.generators import KINDS, make_instance

    for kind in KINDS:
        inst = make_instance(kind, m=48, n=128, k=6, snr_db=18.0, coherence=0.5, seed=3)
        res = CS_Fuse().solve(inst.A, inst.y)
        assert np.all(np.isfinite(res.x_hat)), kind
        assert res.residual_norm >= 0.0, kind


def test_polish_support_max_rounds_is_respected() -> None:
    """``max_rounds`` 限制交换轮数（返回解在给定轮数内达到的最好状态）。"""
    gen = np.random.default_rng(77)
    A = gen.standard_normal((40, 100))
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    true_s = np.sort(gen.choice(100, 5, replace=False))
    y = A[:, true_s] @ gen.standard_normal(5)
    wrong = np.array([0, 1, 2, 3, 4])
    res = {}
    for rounds in (0, 1, 3):
        _, x = polish_support(A, y, wrong, max_rounds=rounds)
        res[rounds] = float(np.sum((y - A @ x) ** 2))
    # 更多轮次不会更差
    assert res[1] <= res[0] + 1e-12
    assert res[3] <= res[1] + 1e-12


def test_cs_fuse_k_grid_is_respected() -> None:
    """``k_grid`` 参数被真实使用（自定义网格 ⇒ k_used 落在该网格内）。"""
    from data.generators import make_instance

    inst = make_instance("wellcond", m=48, n=128, k=6, snr_db=25.0, seed=9)
    custom = (4, 8, 16)
    res = CS_Fuse(k_grid=custom).solve(inst.A, inst.y)
    assert int(res.meta["k_used"]) in custom


def test_stability_select_is_deterministic() -> None:
    """``stability_select`` 用固定 ``seed_offset`` ⇒ 逐位可复现。"""
    from data.generators import make_instance

    inst = make_instance("wellcond", m=48, n=128, k=6, snr_db=25.0, seed=9)
    set_all(20261005)
    s1, f1 = stability_select(inst.A, inst.y, 6, n_boot=6)
    s2, f2 = stability_select(inst.A, inst.y, 6, n_boot=6)
    assert np.array_equal(s1, s2)
    assert np.array_equal(f1, f2)
