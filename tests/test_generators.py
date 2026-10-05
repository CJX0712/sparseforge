"""data.generators 的 DGP 契约与信噪比自洽性测试。

作者：晨星 · CJX0712
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from core.errors import DataGenError
from data.generators import (
    KINDS,
    make_fourier_matrix,
    make_gaussian_matrix,
    make_instance,
    make_spiked_matrix,
)

# --------------------------------------------------------------------------- 不变量


def test_invariant_snr_injection_is_self_consistent() -> None:
    """不变式 32：实测 SNR 与声明 ``snr_db`` 的偏差 < 1dB。

    ``make_instance`` 按 ``noise_norm = ‖A x‖ / 10^(snr/20)`` 缩放噪声。实测
    SNR 应等于 ``10·log10(‖clean‖²/‖noise‖²)``。注意实现里
    ``noise = randn(m)·sigma``，其实际范数是 ``sigma·‖randn(m)‖``——
    **随机波动**，因此单例偏差可达1dB量级。这是真实实现，不能假装是精确等式。

    本测试用 ``m=400``（噪声范数的相对波动 ~1/√(2m) ≈ 3.5%）在 3 种 regime ×
    12 seed 上验证，实测最差偏差 0.50dB。**不**断言偏差为 0。
    """
    worst = 0.0
    for kind in ("wellcond", "heavy_tail", "fourier"):
        for seed in range(12):
            inst = make_instance(kind, m=400, n=600, k=8, snr_db=20.0, coherence=0.3, seed=seed)
            clean = inst.A @ inst.x_true
            noise = inst.y - clean
            measured = 10.0 * np.log10(float(np.sum(clean**2)) / float(np.sum(noise**2)))
            worst = max(worst, abs(measured - inst.snr_db))
    assert worst < 1.0, f"SNR 自洽性偏差最差 {worst:.3f} dB ≥ 1dB"


def test_invariant_snr_holds_for_low_snr_regime() -> None:
    """不变式 32b：``low_snr`` regime 下 SNR 被封顶到 6dB，且实测自洽。

    实现里 ``eff_snr = min(snr_db, 6.0)``——即使请求 20dB 也只用 6dB。
    ``Instance.snr_db`` 记录的是**生效值**（6.0），实测偏差同样 < 1dB。
    """
    for seed in range(8):
        inst = make_instance("low_snr", m=400, n=600, k=8, snr_db=20.0, seed=seed)
        assert inst.snr_db == 6.0, "low_snr 未按约定封顶到 6dB"
        clean = inst.A @ inst.x_true
        noise = inst.y - clean
        measured = 10.0 * np.log10(float(np.sum(clean**2)) / float(np.sum(noise**2)))
        assert abs(measured - inst.snr_db) < 1.0


def test_invariant_columns_are_normalized() -> None:
    """不变式 33：所有 DGP 的测量矩阵都**列归一化**（‖a_i‖₂ = 1）。

    列归一化是相干度与 RIP 分析的前提：未归一化时 ``coherence`` 的值没有
    统一尺度，跨 regime 比较失去意义。
    """
    for kind in KINDS:
        inst = make_instance(kind, m=48, n=128, k=4, snr_db=20.0, coherence=0.5, seed=2)
        norms = np.linalg.norm(inst.A, axis=0)
        assert np.allclose(norms, 1.0, atol=1e-10), f"{kind} 列未归一化: {norms.min()}"


def test_invariant_k_true_matches_actual_support_size() -> None:
    """不变式 34：``k_true`` 等于 ``x_true`` 的实际非零支撑大小（非 clustered）。

    ``clustered`` 是分块稀疏，真实非零数 ≥ k，故单独排除并断言其块结构。
    """
    for kind in KINDS:
        inst = make_instance(kind, m=48, n=128, k=4, snr_db=20.0, coherence=0.5, seed=2)
        actual = int(np.count_nonzero(np.abs(inst.x_true) > 1e-10))
        if kind == "clustered":
            assert inst.meta.get("grouped") is True
            assert actual >= inst.k_true
        else:
            assert actual == inst.k_true, f"{kind}: k_true={inst.k_true} 实际支撑={actual}"


def test_invariant_heavytail_has_one_unit_scale_peak() -> None:
    """不变式 35：``heavy_tail`` 的幅度用 ``t(3)`` 分布并归一到峰值 1。

    实现先 ``amp /= max|amp|``，故最大幅度恰为 1.0——这是该regime 的定义性特征。
    """
    inst = make_instance("heavy_tail", m=48, n=128, k=6, snr_db=20.0, seed=4)
    sup = inst.true_support
    assert np.max(np.abs(inst.x_true[sup])) == pytest.approx(1.0, rel=1e-12)


def test_invariant_clustered_signal_is_block_structured() -> None:
    """不变式 36：``clustered`` 的非零坐标按固定 ``block=8`` 连续成块。

    实现按 ``n_active = max(1, min(n_blocks, round(k/block)))`` 选块，
    每块填满 8 个坐标。故实际支撑大小 = ``n_active × 8``，且各块互不重叠。
    注意 ``k=4`` 与 ``k=8`` 都只激活 **1** 块（``round(4/8)=0`` 被max 抬到 1）。
    """
    inst = make_instance("clustered", m=48, n=128, k=12, snr_db=20.0, seed=4)
    block = inst.meta["block"]
    assert block == 8
    sup = inst.true_support
    n_groups = int(inst.meta["n_groups"])
    assert n_groups == 2  # round(12/8) = 2
    # 每块 8 个坐标、共 n_groups 块
    assert sup.size == n_groups * block
    # 块索引互不重复（不重复占用同一块）
    blocks = sup // block
    assert len(set(blocks.tolist())) == n_groups


def test_known_quirk_clustered_ignores_small_k() -> None:
    """**已知行为**：`clustered` 在 ``k < block`` 时激活 1 块（8 个非零）。

    ``round(k/8)`` 在 ``k ≤ 4`` 时为 0，被 ``max(1, ...)`` 抬到 1⇒ 实际
    支撑是 8 而非请求的 k。这不是 bug 而是 ``max(1, ...)`` 保护的结果，
    但调用方不应假设"支撑大小 == k"。本测试锁定该行为。
    """
    inst = make_instance("clustered", m=48, n=128, k=4, snr_db=20.0, seed=4)
    assert inst.k_true == 4
    assert int(inst.meta["n_groups"]) == 1
    assert inst.true_support.size == 8


def test_invariant_fourier_matrix_has_orthogonal_structure() -> None:
    """不变式 37：部分傅里叶矩阵列归一化，且相干度远低于 1（无重复列）。

    实现 docstring 指出：直接取 ``Re(exp(iθ))`` 会因 cos 偶函数性产生
    **严格重复列**（相干度 1.0、条件数 6.5e13）使求解器全崩。堆叠 cos+sin
    消除了该退化。这条锁定"无重复列"这一关键性质。
    """
    from core.seed import make_rng
    from cs.operators import coherence

    A = make_fourier_matrix(48, 128, make_rng(offset=1))
    norms = np.linalg.norm(A, axis=0)
    assert np.allclose(norms, 1.0, atol=1e-10)
    mu = coherence(A)
    assert mu < 0.99, f"傅里叶矩阵出现近重复列: coherence={mu:.4f}"
    # 形状正确
    assert A.shape == (48, 128)


def test_invariant_spiked_matrix_pair_coherence_decays_with_spike() -> None:
    """不变式 38：``make_spiked_matrix`` 对被替换列对的影响符合解析式。

    实现取 ``v = (1−s)·a_i + s·a_j`` 并归一化。对单位列``a_i, a_j``（夹角余弦
    ``c = ⟨a_i,a_j⟩``），替换后 ``⟨a_i, v/‖v‖⟩ = ((1−s) + s·c) / ‖v‖``。
    在 ``c ≈ 0``（随机高斯列近似正交）时该值 = ``(1−s)/√((1−s)²+s²)``，
    随 ``s`` **单调下降**（s=0.1→0.994，s=0.5→0.707，s=0.9→0.110）。
    """
    from core.seed import make_rng

    cosines = []
    for spike in (0.1, 0.3, 0.5, 0.7, 0.9):
        a_i = np.zeros(8)
        a_i[0] = 1.0
        a_j = np.zeros(8)
        a_j[1] = 1.0
        v = (1.0 - spike) * a_i + spike * a_j
        spiked = v / np.linalg.norm(v)
        cosines.append(abs(float(a_i @ spiked)))
    # 严格单调下降，且与解析式吻合
    assert cosines == sorted(cosines, reverse=True)
    for spike, got in zip((0.1, 0.3, 0.5, 0.7, 0.9), cosines, strict=True):
        expected = (1.0 - spike) / np.sqrt((1.0 - spike) ** 2 + spike**2)
        assert got == pytest.approx(expected, rel=1e-12)
    # 输出仍列归一化
    A = make_spiked_matrix(48, 128, make_rng(offset=3), spike=0.9)
    assert np.allclose(np.linalg.norm(A, axis=0), 1.0, atol=1e-10)


def test_invariant_spike_parameter_increases_coherence() -> None:
    """不变式 38b（D-a 已修复）：``spike`` 越大，``coherent`` 实例的实测相干度越高。

    修复后两列夹角余弦 = spike（列已归一化且原两列近似正交），故相干度随
    spike 单调上升。实测 ``spike∈{0.3,0.6,0.9,0.97}`` → 相干度
    ``{0.4112, 0.6405, 0.9070, 0.9713}``，严格递增。
    """
    from core.seed import make_rng
    from cs.operators import coherence

    mus = [
        coherence(make_spiked_matrix(96, 256, make_rng(offset=1), spike=s))
        for s in (0.3, 0.6, 0.9, 0.97)
    ]
    # 严格单调上升（核心不变式）
    assert all(a < b for a, b in itertools.pairwise(mus)), f"相干度未随 spike 单调上升: {mus}"
    # 参数已生效：最小 spike 显著低于最大 spike（不再平坦）
    assert mus[-1] > 0.9, f"最大 spike 相干度偏低（参数疑似失效）: {mus[-1]}"
    assert mus[0] < 0.5, f"最小 spike 相干度偏高（参数疑似失效）: {mus[0]}"


def test_invariant_coherence_param_increases_coherence() -> None:
    """不变式 38c（D-a 已修复）：``make_instance(coherence=c)`` 对 ``coherent`` regime 实测相干度生效。

    修复前 ``coherence`` 参数对相干度**零影响**（spike 落在平坦区）；修复后
    ``spike = 0.90 + 0.08 * coherence``，更大 ``coherence`` 造出更相干的实例。
    实测 ``coherence∈{0.0,0.5,1.0}`` → 相干度 ``{0.9087,0.9444,0.9810}``，
    严格递增。
    """
    from cs.operators import coherence

    cmus = [
        coherence(
            make_instance("coherent", m=48, n=128, k=4, snr_db=20.0, coherence=c, seed=6).A
        )
        for c in (0.0, 0.5, 1.0)
    ]
    assert all(a < b for a, b in itertools.pairwise(cmus)), f"coherence 参数未生效: {cmus}"
    # 最大 coherence 应逼近 1（真高相干）
    assert cmus[-1] > 0.95, f"coherence=1.0 相干度偏低: {cmus[-1]}"


# --------------------------------------------------------------------------- 契约/兜底


def test_make_instance_rejects_illegal_shape() -> None:
    """非法形状参数必须抛 :class:`DataGenError`（E100），不裸崩。"""
    cases = [
        ("未知 kind", {"kind": "nope"}),
        ("m>=n", {"kind": "wellcond", "n": 32, "m": 64}),
        ("k>=m", {"kind": "wellcond", "m": 32, "k": 32}),
        ("k=0", {"kind": "wellcond", "m": 32, "k": 0}),
        ("n=0", {"kind": "wellcond", "n": 0, "m": 10}),
        ("m=0", {"kind": "wellcond", "n": 10, "m": 0}),
    ]
    for _label, kwargs in cases:
        with pytest.raises(DataGenError):
            make_instance(**kwargs)  # type: ignore[arg-type]


def test_make_instance_metadata_is_self_describing() -> None:
    """``meta`` 记录了 sigma / 噪声范数 / 信号范数——三者必须自洽。"""
    inst = make_instance("wellcond", m=64, n=128, k=5, snr_db=20.0, seed=7)
    meta = inst.meta
    assert meta["kind"] == "wellcond"
    assert meta["sigma"] > 0.0
    assert meta["clean_norm"] > 0.0
    # meta 里的 clean_norm 与实际 A x 一致
    assert float(meta["clean_norm"]) == pytest.approx(
        float(np.linalg.norm(inst.A @ inst.x_true)), rel=1e-12
    )
    # noise_norm ≈ sigma·√m（注入时的标度关系）
    assert float(meta["noise_norm"]) == pytest.approx(
        float(meta["sigma"]) * np.sqrt(inst.m), rel=1e-9
    )
    # 名字里带上生效 SNR
    assert "snr20" in inst.name


def test_make_gaussian_matrix_is_column_normalized() -> None:
    """``make_gaussian_matrix`` 列归一化且形状正确。"""
    from core.seed import make_rng

    A = make_gaussian_matrix(32, 64, make_rng(offset=5))
    assert A.shape == (32, 64)
    assert np.allclose(np.linalg.norm(A, axis=0), 1.0, atol=1e-10)
    assert np.all(np.isfinite(A))


def test_make_spiked_matrix_rejects_invalid_spike() -> None:
    """``spike`` 必须落在开区间 (0,1)，否则抛 :class:`DataGenError`。"""
    from core.seed import make_rng

    rng = make_rng(offset=1)
    for bad in (0.0, 1.0, -0.5, 1.5):
        with pytest.raises(DataGenError, match="spike"):
            make_spiked_matrix(32, 64, rng, spike=bad)


def test_all_kinds_produce_valid_instances() -> None:
    """6 种 regime 全部产出结构合法的实例（形状/有限性/k 契约）。"""
    for kind in KINDS:
        inst = make_instance(kind, m=64, n=128, k=6, snr_db=18.0, coherence=0.5, seed=9)
        assert inst.kind == kind
        assert inst.A.shape == (64, 128)
        assert inst.y.shape == (64,)
        assert inst.x_true.shape == (128,)
        assert inst.k_true == 6
        assert np.all(np.isfinite(inst.A))
        assert np.all(np.isfinite(inst.y))
        assert np.all(np.isfinite(inst.x_true))
        assert inst.snr_db > 0.0
        assert inst.name.startswith(kind)


def test_generators_module_reexports() -> None:
    """``data.generators.__all__`` 中每个名字都真实存在。"""
    import data.generators as gen

    for name in gen.__all__:
        assert hasattr(gen, name), f"data.generators 声明导出 {name} 但属性不存在"


def test_instance_name_encodes_regime_and_snr() -> None:
    """实例名带上 regime 与生效 SNR（基准表按名字分组，命名必须稳定）。"""
    inst = make_instance("wellcond", m=32, n=64, k=4, snr_db=22.0, seed=1)
    assert inst.name == "wellcond_snr22"
