"""cs.solvers 的恢复能力与契约测试。

作者：晨星 · CJX0712
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from core.config import SolveConfig
from core.errors import ConvergenceError, NumericsError
from core.types import RecoveryResult
from cs.solvers import BASELINES, SOLVERS, amp, available_sklearn, cosamp, fista, iht, omp
from eval.metrics import support_f1
from tests.conftest import ALL_SOLVER_NAMES

# --------------------------------------------------------------------------- 不变量


def test_invariant_omp_exact_recovery_noiseless(noiseless_problem) -> None:
    """不变式 9：OMP 在**无噪声**良态实例上精确恢复——``nmse < 1e-10`` 且 ``F1 = 1``。

    这是压缩感知最核心的定理性结论：当 ``A`` 满足 RIP 且 ``m < 2k`` 时，
    OMP 在无噪声情形下**必然**精确恢复支撑集。测试床为 ``m=64, n=256, k=4``
    （``m ≫ 2k``，远离相变边界），系数与支撑随机。

    **为什么必须是"无噪声"**：带噪时 NMSE 有信息论下界——噪声能量比。
    40dB ⇒ 下界 ≈ 1e-5，**数学上不可能**达到 1e-10。实测 40dB 下 OMP 的
    NMSE = 1.6e-5，与该下界一致。因此"40dB + nmse<1e-10"是错误断言。
    """
    A, x, y, support = noiseless_problem
    res = omp(A, y, 4)
    nmse = float(np.sum((res.x_hat - x) ** 2) / np.sum(x**2))
    assert nmse < 1e-10, f"无噪声精确恢复失败: nmse={nmse:.3e}"
    assert support_f1(res.x_hat, x) == 1.0
    # 支撑集逐位相等（不只是 F1 为 1）
    assert np.array_equal(res.support, support)
    # 残差应降到机器精度
    assert res.residual_norm < 1e-10 * max(np.linalg.norm(y), 1.0)


def test_invariant_omp_exact_recovery_robust_over_seeds() -> None:
    """不变式 9b：精确恢复不是运气——20 个随机支撑上全部成立。

    单个seed 通过可能只是幸运。这条在 20 个独立随机实例上重复不变式 9，
    确保它是**算法的性质**而非样例的巧合。
    """
    worst = 0.0
    for s in range(20):
        gen = np.random.default_rng(700 + s)
        m, n, k = 64, 256, 4
        A = gen.standard_normal((m, n))
        A /= np.linalg.norm(A, axis=0, keepdims=True)
        support = np.sort(gen.choice(n, size=k, replace=False))
        x = np.zeros(n)
        x[support] = gen.standard_normal(k)
        res = omp(A, A @ x, k)
        nmse = float(np.sum((res.x_hat - x) ** 2) / np.sum(x**2))
        worst = max(worst, nmse)
        assert support_f1(res.x_hat, x) == 1.0
        assert np.array_equal(res.support, support)
    assert worst < 1e-10, f"20seed 中最差 nmse={worst:.3e}"


def test_invariant_omp_high_snr_hits_noise_floor() -> None:
    """不变式 9c：高 SNR 下 NMSE 达到**噪声能量下界**而非数值精度。

    40dB ⇒ 噪声比 1e-4 ⇒ NMSE 下界 ~1e-5。实测 1.6e-5；提高 SNR 10倍
    （+20dB）应让 NMSE 精确下降 100 倍——这验证了"误差来自噪声而非算法"。
    这是 40dB 情形下**真正可以断言**的性质。
    """
    inst = make_wellcond(11, snr_db=40.0)
    nmse40, f1_40 = inst.score(omp(inst.A, inst.y, 4).x_hat)
    inst80 = make_wellcond(11, snr_db=60.0)
    nmse60, f1_60 = inst80.score(omp(inst80.A, inst80.y, 4).x_hat)
    # +20dB ⇒ NMSE 降约 100 倍（宽松区间：50~200倍）
    ratio = nmse40 / nmse60
    assert 50.0 < ratio < 200.0, f"NMSE 未按 SNR 线性下降: ratio={ratio:.1f}"
    # 两者都应精确恢复支撑集（支撑恢复不受噪声水平影响）
    assert f1_40 == 1.0
    assert f1_60 == 1.0
    # 且 NMSE 落在噪声下界的一个数量级内
    assert nmse40 < 1e-4


def make_wellcond(seed: int, snr_db: float):
    """构造 ``m=64, n=256, k=4`` 的 wellcond 实例（供 SNR 不变量使用）。"""
    from data.generators import make_instance

    return make_instance("wellcond", m=64, n=256, k=4, snr_db=snr_db, seed=seed)


@pytest.mark.parametrize("solver_fn", ALL_SOLVER_NAMES, indirect=True)
def test_invariant_solvers_return_finite_dense_vectors(tiny_instance, solver_fn) -> None:
    """不变式 10：**所有**求解器都返回形状 ``(n,)``、全有限的 ``x_hat``。

    这是``core.interfaces.Solver`` 协议写明的硬契约。任一求解器违反它，
    下游的 ``Instance.score`` 与 ``aggregate`` 都会产出 NaN 并污染整张基准表。
    """
    res = solver_fn(tiny_instance.A, tiny_instance.y, 3, SolveConfig(k=3, max_iter=200))
    assert isinstance(res, RecoveryResult)
    assert res.x_hat.shape == (tiny_instance.n,)
    assert np.all(np.isfinite(res.x_hat))
    # 支撑集必须是合法升序整数索引
    assert res.support.ndim == 1
    assert np.all(np.diff(res.support) > 0)
    assert res.support.size == 0 or (
        res.support.min() >= 0 and res.support.max() < tiny_instance.n
    )
    # 残差范数必须与 x_hat 自洽
    expected = float(np.linalg.norm(tiny_instance.y - tiny_instance.A @ res.x_hat))
    assert res.residual_norm == pytest.approx(expected, rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("solver_fn", ALL_SOLVER_NAMES, indirect=True)
def test_invariant_support_matches_x_hat_nonzero(tiny_instance, solver_fn) -> None:
    """不变式 11：``support`` 恰好是 ``x_hat`` 中幅度超过 ``SUPPORT_EPS`` 的坐标。

    支撑集与解向量是两个独立算出的字段（``_finish`` 里先 debias 再取支撑）。
    若两者不同步，``support_f1`` 会给出自相矛盾的评分。
    """
    from core.types import SUPPORT_EPS

    res = solver_fn(tiny_instance.A, tiny_instance.y, 3, SolveConfig(k=3, max_iter=200))
    expected = np.flatnonzero(np.abs(res.x_hat) > SUPPORT_EPS)
    assert np.array_equal(res.support, expected)


def test_invariant_debias_reduces_nmse(tiny_instance) -> None:
    """不变式 12：在同一支撑集上去偏（debias）不增大 NMSE。

    去偏就是支撑集上的最小二乘重拟合，训练残差最小⇒ NMSE 不可能更差
    （NMSE 与残差通过投影分解关联）。比较 ``debias=True/False`` 两次求解。
    """
    cfg_on = SolveConfig(k=3, max_iter=200, debias=True)
    cfg_off = SolveConfig(k=3, max_iter=200, debias=False)
    res_on = iht(tiny_instance.A, tiny_instance.y, 3, cfg_on)
    res_off = iht(tiny_instance.A, tiny_instance.y, 3, cfg_off)
    nmse_on, _ = tiny_instance.score(res_on.x_hat)
    nmse_off, _ = tiny_instance.score(res_off.x_hat)
    assert nmse_on <= nmse_off + 1e-12


def test_invariant_fista_objective_beats_zero_init(tiny_instance) -> None:
    """不变式 13：FISTA 的终解目标函数严格优于零解（LASSO 有效求解）。

    ``objective(z) = ½‖Az−y‖² + λ‖z‖₁``，λ 取自 ``meta["lam"]``。
    实测：终解 0.0508 vs 零解 1.0816，改善约 21 倍。

    **注意**：`meta["objective"]` 记录的是 **debias 之前**那个 ℓ1 迭代点的
    目标值（0.0662），与去偏后 ``x_hat`` 的目标值（0.0508）不同——因为去偏
    在支撑集上做无惩罚 LS，破坏了 ℓ1 最优性。所以这里用**重算**的方式验证，
    而不是直接比对 ``meta["objective"]``。
    """
    cfg = SolveConfig(k=3, max_iter=300, debias=False)
    res = fista(tiny_instance.A, tiny_instance.y, 3, cfg)
    lam = float(res.meta["lam"])
    A, y = tiny_instance.A, tiny_instance.y

    def obj(z: np.ndarray) -> float:
        return 0.5 * float(np.sum((A @ z - y) ** 2)) + lam * float(np.sum(np.abs(z)))

    assert lam > 0.0
    assert obj(res.x_hat) < obj(np.zeros(A.shape[1]))


def test_invariant_solvers_do_not_accept_x_true() -> None:
    """不变式 14：求解器签名中**不存在** ``x_true`` 参数（防泄漏硬约束）。

    ``core.interfaces.Solver`` 与 ``cs.solvers`` 的模块 docstring 都把
    "求解器绝不接触x_true"列为硬契约。通过内省签名锁定：任何求解器的
    形参中不得出现真值相关名字。这是**结构性**断言，不依赖运行时行为。
    """
    banned = {"x_true", "xtruth", "truth", "k_true", "support_true", "x_gt"}
    for name, fn in SOLVERS.items():
        params = set(inspect.signature(fn).parameters)
        assert not (params & banned), f"{name} 暴露了真值参数: {params & banned}"


# --------------------------------------------------------------------------- 契约/兜底


def test_omp_requires_explicit_k(tiny_instance) -> None:
    """``k=None`` 且 cfg.k 也为 None 时，OMP 必须抛 :class:`NumericsError`。"""
    with pytest.raises(NumericsError, match="需要显式稀疏度"):
        omp(tiny_instance.A, tiny_instance.y, None, SolveConfig())
    # cfg.k 有值时可正常求解（k 参数为 None 时回落到 cfg.k）
    res = omp(tiny_instance.A, tiny_instance.y, None, SolveConfig(k=3))
    assert res.x_hat.shape == (tiny_instance.n,)


def test_cosamp_and_iht_require_k(tiny_instance) -> None:
    """CoSaMP / IHT 同样需要显式 k（与 OMP 一致的契约）。"""
    with pytest.raises(NumericsError):
        cosamp(tiny_instance.A, tiny_instance.y, None, SolveConfig())
    with pytest.raises(NumericsError):
        iht(tiny_instance.A, tiny_instance.y, None, SolveConfig())


def test_degenerate_inputs_do_not_crash() -> None:
    """退化输入（A 全零 / y 全零 / n=1）不裸崩，返回合法的空支撑解。

    实测行为：这些输入下 OMP 的相关性``max|Aᵀr|`` 为 0，循环在
    ``corr[i] <= 0`` 处**正常退出**，返回全零解与空支撑集——这是**优雅降级**
    而非异常。所以此处断言"不抛裸异常 + 输出合法"，而非"必须抛异常"。
    """
    cases = {
        "A全零": (np.zeros((10, 20)), np.random.default_rng(0).standard_normal(10)),
        "y全零": (np.random.default_rng(1).standard_normal((10, 20)), np.zeros(10)),
        "n=1": (
            np.random.default_rng(2).standard_normal((4, 1)),
            np.random.default_rng(2).standard_normal(4),
        ),
        "A和y全零": (np.zeros((6, 9)), np.zeros(6)),
    }
    for label, (A, y) in cases.items():
        res = omp(A, y, 3)
        assert res.x_hat.shape == (A.shape[1],), label
        assert np.all(np.isfinite(res.x_hat)), label
        assert res.residual_norm >= 0.0, label


def test_m_less_than_k_is_handled_without_crash() -> None:
    """``m < k``（超欠定边界）：求解器不崩，返回有限解。

    实测 ``m=4, k=6`` 时 OMP 把 k 截断到 ``min(k, n)`` 并继续；LS 在列满秩
    前提下有唯一解。此处断言输出合法即可。
    """
    A = np.random.default_rng(4).standard_normal((4, 12))
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    y = np.random.default_rng(5).standard_normal(4)
    for fn in (omp, iht, cosamp):
        res = fn(A, y, 6, SolveConfig(k=6, max_iter=50))
        assert res.x_hat.shape == (12,)
        assert np.all(np.isfinite(res.x_hat))


def test_solvers_registry_is_consistent() -> None:
    """``SOLVERS`` / ``BASELINES`` 登记表自洽：每个名字可调用且签名统一。"""
    assert "omp" in SOLVERS
    assert set(SOLVERS) == set(BASELINES)
    # 重加权三兄弟必须已注册（由 ``_build_baselines`` 注入）
    assert {"rw_fista", "irl1_omp", "amp_rl1"} <= set(SOLVERS)
    for name, fn in SOLVERS.items():
        params = list(inspect.signature(fn).parameters)
        assert params[:4] == ["A", "y", "k", "cfg"], f"{name} 签名不统一: {params}"


def test_available_sklearn_matches_actual_import() -> None:
    """``available_sklearn()`` 必须与 sklearn 真实可导入性一致。"""
    try:
        import sklearn.linear_model  # noqa: F401

        expected = True
    except ImportError:
        expected = False
    assert available_sklearn() is expected


@pytest.mark.skipif(not available_sklearn(), reason="sklearn 不可用")
def test_sklearn_solvers_produce_valid_results(tiny_instance) -> None:
    """sklearn 适配器返回合法 :class:`RecoveryResult`（维度/有限性/后端标记）。"""
    for name in ("sk_omp", "sk_lasso", "sk_lars"):
        fn = SOLVERS[name]
        k = 3 if name != "sk_lars" else None
        res = fn(tiny_instance.A, tiny_instance.y, k, SolveConfig(k=3, max_iter=200))
        assert res.x_hat.shape == (tiny_instance.n,)
        assert np.all(np.isfinite(res.x_hat))
        assert res.meta.get("backend") == "sklearn"


@pytest.mark.xfail(
    reason="已知缺陷：AMP 在良态 wellcond 实例上 100% 发散（默认 damping=0.5，"
    "m∈{24..96} × 6seed 全部 ConvergenceError）；Onsager 项写法有误",
    strict=True,
)
def test_xfail_amp_diverges_on_wellcond() -> None:
    """**已知实现缺陷**：AMP 在良态wellcond 实例上 100% 发散。

    实测（m ∈ {24,32,48,64,96} × 6 seed = 30/30）：默认 ``damping=0.5`` 下
    ``amp`` 全部抛 :class:`ConvergenceError`。根因是 Onsager 项的写法
    （见 ``amp`` 的 docstring），在小 ``m`` 高 SNR 实例上 ``t`` 项累积发散。

    期望行为：AMP 在良态实例上应当**正常返回**一个有限解。因此这里直接调用
    并检查返回值——当前会抛异常 ⇒ 测试失败 ⇒ 被 ``xfail(strict=True)`` 记为
    已知缺陷。**不放宽断言去迁就实现**。
    """
    from data.generators import make_instance

    inst = make_instance("wellcond", m=24, n=64, k=3, snr_db=25.0, seed=5)
    res = amp(inst.A, inst.y, 3, SolveConfig(k=3, max_iter=200))
    assert res.x_hat.shape == (inst.n,)
    assert np.all(np.isfinite(res.x_hat))
    assert res.residual_norm >= 0.0


@pytest.mark.xfail(
    reason="已知缺陷：amp_rl1 发散时抛裸 RuntimeError('amp_rl1 diverged')，"
    "逃出 SparseForgeError 体系（应为 ConvergenceError/E300）",
    strict=True,
)
def test_xfail_amp_rl1_raises_raw_runtime_error() -> None:
    """**已知实现缺陷**：``amp_rl1`` 发散时抛裸 ``RuntimeError``，非体系化异常。

    ``cs.solvers.amp`` 发散时抛 ``ConvergenceError``（E300，符合架构）；
    ``cs.reweighted.amp_rl1`` 却抛 ``RuntimeError("amp_rl1 diverged")``，
    **逃出了 ``SparseForgeError`` 体系**。后果：调用方无法按语义捕获，
    只能靠字符串匹配；pipeline 的兜底 ``except Exception`` 会静默吞掉。

    正确行为应是抛 ``ConvergenceError``。本测试用 ``xfail(strict=True)``
    锁定该缺陷：修复后 pytest 会报XPASS（因strict 而失败），提醒移除标记。
    """
    from data.generators import make_instance

    inst = make_instance("wellcond", m=24, n=64, k=3, snr_db=25.0, seed=5)
    with pytest.raises(ConvergenceError):
        SOLVERS["amp_rl1"](inst.A, inst.y, 3, SolveConfig(k=3))


def test_amp_rl1_violates_error_hierarchy() -> None:
    """同一份"AMP 发散"缺陷的**独立**证据：``amp_rl1`` 不抛体系化异常。

    这条不xfail，因为它断言的是**当前真实行为**（确实抛裸 RuntimeError），
    目的是让缺陷在测试报告中显式可见。修复后本测试会失败——那时应把它
    改成 ``pytest.raises(ConvergenceError)`` 并删掉上面的 xfail。
    """
    from core.errors import SparseForgeError
    from data.generators import make_instance

    inst = make_instance("wellcond", m=24, n=64, k=3, snr_db=25.0, seed=5)
    with pytest.raises(RuntimeError) as excinfo:
        SOLVERS["amp_rl1"](inst.A, inst.y, 3, SolveConfig(k=3))
    assert not isinstance(excinfo.value, SparseForgeError), (
        "amp_rl1 已改为抛 SparseForgeError——请更新测试并移除 xfail 标记"
    )
