"""eval.metrics 的指标语义与聚合守恒测试。

作者：晨星 · CJX0712
"""

from __future__ import annotations

import numpy as np
import pytest

from core.types import SUPPORT_EPS, BenchmarkRow
from eval.metrics import aggregate, beats, mean_std, nmse, success_of, support_f1

# --------------------------------------------------------------------------- 不变量


def test_invariant_beats_matches_its_definition() -> None:
    """不变式 15：``beats`` 严格实现 ``a<b 且 b−a > factor·(σ_a+σ_b)``。

    逐条覆盖真值表的边界：显著性足够 ⇒ True；差距不足 ⇒ False；
    ``a`` 更差 ⇒ False；``σ`` 全零时退化为纯均值比较；``factor`` 放大到
    超过实际差距 ⇒ False。
    """
    # 差距 0.5 > 0.5·(0.1+0.1)=0.1 ⇒ 显著 ⇒ True
    assert beats(0.5, 0.1, 1.0, 0.1, 0.5) is True
    # 差距 0.01 ≮ 0.1 ⇒ 不显著 ⇒ False
    assert beats(0.99, 0.1, 1.0, 0.1, 0.5) is False
    # σ 全零 ⇒ denom<=0 ⇒ 退化为纯均值比较 ⇒ True
    assert beats(0.5, 0.0, 1.0, 0.0, 0.5) is True
    # a 比 b 差（a_mean > b_mean）⇒ 恒 False，无论方差多大
    assert beats(1.0, 0.1, 0.5, 0.1, 0.5) is False
    # factor 放大到 5.0 ⇒ 阈值 1.0 > 实际差距 0.5 ⇒ False
    assert beats(0.5, 0.1, 1.0, 0.1, 5.0) is False
    # 均值相等 ⇒ 不满足 a<b ⇒ False
    assert beats(1.0, 0.1, 1.0, 0.1, 0.5) is False


def test_invariant_beats_boundary_is_strict() -> None:
    """不变式 15b：判据是**严格**不等式——恰好等于阈值时返回 False。

    ``beats`` 用 ``>`` 而非 ``>=``，故 ``b−a == factor·(σ_a+σ_b)` 是 False。
    这排除了"实现偷偷用了 ``>=``"的可能。
    """
    # b−a = 0.2, factor·(σ_a+σ_b) = 0.5·0.4 = 0.2 ⇒ 恰好相等 ⇒ False
    assert beats(0.8, 0.2, 1.0, 0.2, 0.5) is False
    # 略大一点 ⇒ True
    assert beats(0.79, 0.2, 1.0, 0.2, 0.5) is True


def test_invariant_aggregate_mean_is_arithmetic_mean() -> None:
    """不变式 16：``aggregate`` 的 ``nmse_mean`` 等于手算算术平均（相对误差 < 1e-12）。

    ``aggregate`` 是基准表的口径入口，门禁（GATE-1/2/4）全部基于它的输出。
    若聚合口径与手算不一致，所有门禁结论都不可信。
    """
    rows = make_rows(
        [
            ("i1", 1, "omp", 0.1),
            ("i1", 2, "omp", 0.3),
            ("i2", 1, "omp", 0.2),
            ("i2", 2, "omp", 0.4),
        ]
    )
    agg = aggregate(rows)
    manual = (0.1 + 0.3 + 0.2 + 0.4) / 4
    got = float(agg["omp"]["nmse_mean"])
    assert abs(got - manual) / manual < 1e-12
    assert got == pytest.approx(manual, rel=1e-12)


def test_invariant_aggregate_std_is_sample_std() -> None:
    """不变式 16b：``nmse_std`` 是**样本**标准差（``ddof=1``），非总体标准差。

    ``mean_std`` 显式用 ``ddof=1``。基准约定 n_seeds ≥ 3 正是为了让样本
    std 有意义；若误用 ``ddof=0``，GATE-2 的显著性判定会系统性偏严。
    """
    vals = [0.1, 0.3, 0.2, 0.4]
    rows = make_rows([("i1", i, "omp", v) for i, v in enumerate(vals)])
    agg = aggregate(rows)
    got = float(agg["omp"]["nmse_std"])
    assert got == pytest.approx(float(np.std(vals, ddof=1)), rel=1e-12)
    # 与总体标准差显著不同（n=4 时二者差因子 sqrt(4/3)）
    assert abs(got - float(np.std(vals, ddof=0))) > 1e-6


def test_invariant_aggregate_conserves_row_count() -> None:
    """不变式 16c：``aggregate`` 不丢行——``n`` 等于输入总行数，``per_instance`` 求和也相等。"""
    specs = [
        ("i1", 1, "omp", 0.1),
        ("i1", 2, "omp", 0.3),
        ("i2", 1, "omp", 0.2),
        ("i3", 1, "omp", 0.5),
        ("i1", 1, "cosamp", 0.4),
        ("i2", 1, "cosamp", 0.6),
    ]
    rows = make_rows(specs)
    agg = aggregate(rows)
    assert agg["omp"]["n"] == 4
    assert agg["cosamp"]["n"] == 2
    for method in ("omp", "cosamp"):
        per = agg[method]["per_instance"]
        assert sum(int(v["n"]) for v in per.values()) == int(agg[method]["n"])


def test_invariant_support_f1_extremes() -> None:
    """不变式 17：``support_f1`` 的三个端点值精确：全对=1.0、全错=0.0、双空=1.0。

    F1 = 2·prec·rec/(prec+rec)。全对 ⇒ prec=rec=1 ⇒ 1.0；无交集 ⇒ 0.0；
    两边都空 ⇒ 约定为 1.0（"都判断为无信号"算完美一致）。
    """
    assert support_f1(np.array([1.0, 0, 0]), np.array([1.0, 0, 0])) == 1.0
    assert support_f1(np.array([1.0, 0, 0]), np.array([0.0, 1.0, 0])) == 0.0
    assert support_f1(np.zeros(5), np.zeros(5)) == 1.0
    # 一方为空、另一方非空 ⇒ 0.0
    assert support_f1(np.zeros(5), np.array([1.0, 0, 0, 0, 0])) == 0.0


def test_invariant_support_f1_matches_definition() -> None:
    """不变式 17b：``support_f1`` 与 F1 定义式逐例一致（含部分命中）。"""
    x_hat = np.array([1.0, 1.0, 0.0, 0.0])
    x_true = np.array([1.0, 0.0, 0.0, 0.0])
    pred = {0, 1}
    true = {0}
    inter = len(pred & true)
    prec = inter / len(pred)
    rec = inter / len(true)
    expected = 2 * prec * rec / (prec + rec)
    assert support_f1(x_hat, x_true) == pytest.approx(expected, rel=1e-15)
    assert support_f1(x_hat, x_true) == pytest.approx(2 / 3, rel=1e-12)


def test_invariant_nmse_is_scale_invariant() -> None:
    """不变式 18：``nmse`` 对真值与估计的**同比例缩放**不变。

    ``nmse = ‖x̂−x‖²/‖x‖²``：分子分母同乘 ``c²`` 后约掉。这是 NMSE 作为
    *相对*误差的核心意义——不同幅度的信号可以横向比较。
    """
    gen = np.random.default_rng(17)
    x = gen.standard_normal(40)
    x_hat = x + 0.1 * gen.standard_normal(40)
    base = nmse(x_hat, x)
    for c in (0.01, 2.0, 100.0, 1e6):
        assert nmse(c * x_hat, c * x) == pytest.approx(base, rel=1e-12)


def test_invariant_nmse_denominator_is_x_true_norm() -> None:
    """不变式 18b：``nmse`` 的**分母是 ``‖x_true‖²``**——它刻意**不是**对称的。

    ``nmse(x_hat, x_true) = ‖x_hat − x_true‖² / ‖x_true‖²``。分子对称、分母
    只取真值 ⇒ ``nmse(a,b) ≠ nmse(b,a)``（实测 2.05 vs 2.91）。这条锁定参数
    顺序的语义：分母必须随第二个参数（真值）走。若实现误用 ``‖x_hat‖²``
    做分母，或把两参数写反，误差就会被静默归一化到错误尺度。

    对应的正确不变量是"分母跟随真值"：把真值缩放 ``c`` 倍会让NMSE 变``1/c²``。
    """
    gen = np.random.default_rng(19)
    a = gen.standard_normal(25)
    b = gen.standard_normal(25)
    base = nmse(a, b)
    # 定义式逐位吻合：分子 ‖a−b‖²、分母 ‖b‖²（第二个参数 = 真值）
    assert base == pytest.approx(float(np.sum((a - b) ** 2) / np.sum(b**2)), rel=1e-15)
    # 分母取x_true 而非 x_hat：两个参数**范数不同**时结果必须与
    # "用 x_hat 归一化"的那个值不同（否则说明分母写错了对象）
    other = float(np.sum((a - b) ** 2) / np.sum(a**2))
    assert not np.isclose(base, other, rtol=1e-6), "两个范数恰好相近，换个用例"
    assert base == pytest.approx(other * float(np.sum(a**2) / np.sum(b**2)), rel=1e-12)
    # 只缩放 x_true 时分子分母同时变，**不**存在 1/c² 律（避免误写断言）
    assert not np.isclose(nmse(a, 2.0 * b), base / 4.0, rtol=1e-6)


def test_invariant_success_of_threshold() -> None:
    """不变式 19：``success_of(f1)`` 当且仅当 ``f1 ≥ 1 − tol`` 时为 True。

    默认 ``tol=1e-9``，故 0.999 **不算**成功。这条锁定判据的严格性。
    """
    assert success_of(1.0) is True
    assert success_of(1.0 - 1e-12) is True
    assert success_of(0.999) is False
    assert success_of(0.0) is False
    # 自定义容差生效
    assert success_of(0.999, tol=1e-2) is True


def test_invariant_support_eps_threshold_semantics() -> None:
    """不变式 19b：支撑判定忽略幅度 ≤ ``SUPPORT_EPS``(=1e-10) 的分量。

    低于该阈值的坐标视为"零"。这条确保数值噪声不会虚增支撑集大小。
    """
    eps = SUPPORT_EPS
    x = np.array([1.0, eps * 0.5, -eps * 0.5, eps * 2.0])
    # 只有 index 0 和 3 超过阈值（index 1/2 幅度 0.5e-10 被当作零）
    assert np.array_equal(np.flatnonzero(np.abs(x) > eps), np.array([0, 3]))
    # 对真值全1 的向量：pred={0,3}、true={0,1,2,3}、inter={0,3}
    # ⇒ prec=2/2=1、rec=2/4=0.5、F1=2·1·0.5/1.5=2/3
    y = np.ones(4)
    pred, true, inter = {0, 3}, {0, 1, 2, 3}, 2
    prec, rec = inter / len(pred), inter / len(true)
    assert support_f1(x, y) == pytest.approx(2 * prec * rec / (prec + rec), rel=1e-12)
    assert support_f1(x, y) == pytest.approx(2 / 3, rel=1e-12)


# --------------------------------------------------------------------------- 单元/边界


def test_mean_std_edge_cases() -> None:
    """``mean_std``：空序列 ⇒ (0,0)；单元素 ⇒ (v,0)；否则样本std。"""
    assert mean_std([]) == (0.0, 0.0)
    assert mean_std([5.0]) == (5.0, 0.0)
    assert mean_std([1.0, 2.0, 3.0]) == (2.0, 1.0)
    # 接受 ndarray
    assert mean_std(np.array([1.0, 2.0, 3.0])) == (2.0, 1.0)


def test_nmse_zero_denominator_convention() -> None:
    """``nmse`` 在 ``x_true`` 全零时约定返回 0.0（而非 ZeroDivisionError）。"""
    assert nmse(np.ones(4), np.zeros(4)) == 0.0


def test_aggregate_on_empty_rows() -> None:
    """空输入返回空字典（不抛异常）。"""
    assert aggregate([]) == {}


def test_aggregate_success_rate_is_fraction() -> None:
    """``success_rate`` = 完全恢复支撑集的实例占比。"""
    rows = make_rows(
        [("i1", 1, "omp", 0.1), ("i1", 2, "omp", 0.2), ("i1", 3, "omp", 0.3)],
        f1_values=[1.0, 1.0, 0.5],
    )
    agg = aggregate(rows)
    assert float(agg["omp"]["success_rate"]) == pytest.approx(2 / 3, rel=1e-12)


def test_benchmark_row_to_dict_roundtrip() -> None:
    """:class:`BenchmarkRow.to_dict` 字段齐全且类型正确（aggregate 的输入契约）。"""
    row = BenchmarkRow(
        instance="i",
        seed=7,
        method="omp",
        nmse=0.1,
        support_f1=1.0,
        success=True,
        residual_norm=0.2,
        n_iter=5,
        elapsed_sec=0.01,
    )
    d = row.to_dict()
    assert set(d) == {
        "instance",
        "seed",
        "method",
        "nmse",
        "support_f1",
        "success",
        "residual_norm",
        "n_iter",
        "elapsed_sec",
        "meta",
    }
    assert isinstance(d["seed"], int)
    assert isinstance(d["success"], bool)
    assert aggregate([d])["omp"]["n"] == 1


def make_rows(specs, f1_values=None) -> list[dict]:
    """从 ``(instance, seed, method, nmse)`` 四元组构造 aggregate 所需的行字典。"""
    rows = []
    for i, (inst, seed, method, val) in enumerate(specs):
        f1 = 1.0 if f1_values is None else f1_values[i]
        rows.append(
            {
                "instance": inst,
                "seed": seed,
                "method": method,
                "nmse": val,
                "support_f1": f1,
                "success": f1 >= 1.0,
                "residual_norm": 0.1,
                "n_iter": 1,
                "elapsed_sec": 0.0,
                "meta": {},
            }
        )
    return rows
