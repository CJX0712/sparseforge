"""异常层级与配置校验测试。

作者：晨星 · CJX0712
"""

from __future__ import annotations

import numpy as np
import pytest

from core.config import BenchmarkConfig, SolveConfig, config_from_env, load_config
from core.errors import (
    BackendError,
    ConfigError,
    ConvergenceError,
    DataGenError,
    NumericsError,
    SparseForgeError,
)
from core.types import GATE_THRESHOLDS, SUCCESS_NMSE, SUPPORT_EPS, Instance, RecoveryResult

_ALL_ERRORS = (DataGenError, NumericsError, ConvergenceError, ConfigError, BackendError)


# --------------------------------------------------------------------------- 不变量


def test_invariant_error_codes_are_stable() -> None:
    """不变式 28：每个异常类型的 ``.code`` 前缀正确且**唯一**。

    错误码是对外契约（CLI 输出、日志聚合、跨服务追踪都依赖它），一旦漂移
    故监控就失效。故断言精确字符串而非"以 E 开头"。
    """
    expected = {
        DataGenError: "E100",
        NumericsError: "E200",
        ConvergenceError: "E300",
        ConfigError: "E400",
        BackendError: "E500",
    }
    seen: list[str] = []
    for cls, code in expected.items():
        assert cls.code == code, f"{cls.__name__}.code 应为 {code}，实为 {cls.code}"
        seen.append(cls.code)
    # 五个码互不重复
    assert len(set(seen)) == len(seen)
    # 基类有自己的码
    assert SparseForgeError.code == "E000"


def test_invariant_errors_all_subclass_base() -> None:
    """不变式 28b：五个语义层异常都继承 :class:`SparseForgeError`。

    这让调用方可以``except SparseForgeError`` 一次捕获全部体系内错误。
    """
    for cls in _ALL_ERRORS:
        assert issubclass(cls, SparseForgeError), f"{cls.__name__} 未继承基类"
    assert issubclass(SparseForgeError, Exception)


def test_invariant_error_message_embeds_code_and_context() -> None:
    """不变式 29：``str(exc)`` 含 ``[code]`` 前缀与 ``| k=v`` 上下文。

    上下文格式化是排障时的主要信息载体（不看 traceback 就能知道是哪个参数、
    哪个值越界）。
    """
    exc = NumericsError("算子输出含 NaN/Inf", shape=(24, 64), solver="omp")
    text = str(exc)
    assert text.startswith("[E200] ")
    assert "算子输出含 NaN/Inf" in text
    assert "| shape=(24, 64)" in text
    assert "| solver=omp" in text
    # context 以 dict 形式可编程访问
    assert exc.context == {"shape": (24, 64), "solver": "omp"}


def test_invariant_config_rejects_illegal_values() -> None:
    """不变式 30：配置校验拒绝非法值——非法配置**必须**抛 :class:`ConfigError`。

    逐一覆盖任务书点名的三类（``coherence=1.5`` / ``refine_rounds=-1`` / ``k > m``）
    以及其余全部校验点。静默接受非法配置会让错误在数十秒的基准跑完后才暴露。
    """
    cases: list[tuple[str, object]] = [
        ("coherence=1.5", lambda: BenchmarkConfig(coherence=1.5)),
        ("coherence=-0.1", lambda: BenchmarkConfig(coherence=-0.1)),
        ("refine_rounds=-1", lambda: BenchmarkConfig(refine_rounds=-1)),
        ("k>=m", lambda: BenchmarkConfig(n=64, m=32, k=32)),
        ("k=0", lambda: BenchmarkConfig(k=0)),
        ("m>=n", lambda: BenchmarkConfig(n=32, m=64)),
        ("n<=0", lambda: BenchmarkConfig(n=0)),
        ("m<=0", lambda: BenchmarkConfig(m=0)),
        ("n_seeds=0", lambda: BenchmarkConfig(n_seeds=0)),
        ("ensemble_size=0", lambda: BenchmarkConfig(ensemble_size=0)),
        ("hpo_trials=0", lambda: BenchmarkConfig(hpo_trials=0)),
        ("tau=0", lambda: BenchmarkConfig(tau=0.0)),
        ("lam_grid 含非正", lambda: BenchmarkConfig(lam_grid=(0.1, -0.2))),
        ("SolveConfig max_iter=0", lambda: SolveConfig(max_iter=0)),
        ("SolveConfig damping=1.5", lambda: SolveConfig(damping=1.5)),
        ("SolveConfig damping=-0.1", lambda: SolveConfig(damping=-0.1)),
        ("SolveConfig tol<0", lambda: SolveConfig(tol=-1.0)),
        ("SolveConfig block_size<0", lambda: SolveConfig(block_size=-1)),
        ("SolveConfig refine_iters<0", lambda: SolveConfig(refine_iters=-1)),
    ]
    for _label, factory in cases:
        with pytest.raises(ConfigError):
            factory()  # type: ignore[operator]


def test_invariant_valid_config_is_accepted() -> None:
    """不变式 30b：合法配置不被误拒（含边界值 damping=0/1、coherence=0）。

    校验器过严同样是 bug——它会让合法的消融配置无法运行。
    """
    BenchmarkConfig()
    BenchmarkConfig(coherence=0.0, refine_rounds=0)
    SolveConfig(damping=0.0)
    SolveConfig(damping=1.0)
    SolveConfig(tol=0.0, max_iter=1)
    SolveConfig(block_size=0, refine_iters=0)


def test_invariant_gate_thresholds_are_positive_and_stable() -> None:
    """不变式 31：预注册门禁阈值为正且处于合理区间（禁止负数/NaN）。"""
    for key, value in GATE_THRESHOLDS.items():
        assert isinstance(value, float), key
        assert value > 0.0, f"{key} 阈值应为正，实为 {value}"
        assert np.isfinite(value), key
    # 方案阶段定死的关键阈值
    assert GATE_THRESHOLDS["g1_vs_weak_rel_gain"] == 0.30
    assert GATE_THRESHOLDS["g2_vs_best_rel_gain"] == 0.20
    assert GATE_THRESHOLDS["g2_sigma_factor"] == 0.5
    assert GATE_THRESHOLDS["g2_noninferiority_ratio"] == 1.10
    assert GATE_THRESHOLDS["g3_ablation_rel_drop"] == 0.02


def test_invariant_support_eps_is_negligible() -> None:
    """不变式 31b：``SUPPORT_EPS`` 远小于信号幅度，``SUCCESS_NMSE`` 为严格判据。"""
    assert 0.0 < SUPPORT_EPS < 1e-8
    assert SUCCESS_NMSE == 1e-8


# --------------------------------------------------------------------------- 异常捕获语义


def test_errors_are_catchable_by_base_class() -> None:
    """五个异常都能被 ``except SparseForgeError`` 捕获（体系有效的证据）。"""
    for cls in _ALL_ERRORS:
        try:
            raise cls("boom", key="value")
        except SparseForgeError as exc:
            assert isinstance(exc, cls)
            assert exc.context == {"key": "value"}
        except Exception:  # pragma: no cover - 不应到达
            pytest.fail(f"{cls.__name__} 未被基类捕获")


def test_error_context_defaults_to_empty_dict() -> None:
    """无上下文时 ``context`` 为空 dict（不是 None，调用方无需判空）。"""
    exc = NumericsError("裸消息")
    assert exc.context == {}
    assert str(exc) == "[E200] 裸消息"


def test_errors_module_reexports() -> None:
    """``core.errors.__all__`` 中每个名字都真实存在。"""
    import core.errors as errors

    for name in errors.__all__:
        assert hasattr(errors, name), f"core.errors 声明导出 {name} 但属性不存在"


# --------------------------------------------------------------------------- 配置/env


def test_config_from_env_overrides_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """``ENV_SPARSEFORGE_<FIELD>`` 覆盖 dataclass 字段。"""
    monkeypatch.setenv("ENV_SPARSEFORGE_N", "128")
    monkeypatch.setenv("ENV_SPARSEFORGE_M", "64")
    monkeypatch.setenv("ENV_SPARSEFORGE_K", "8")
    monkeypatch.setenv("ENV_SPARSEFORGE_SNR_DB", "25.5")
    cfg = load_config()
    assert (cfg.n, cfg.m, cfg.k, cfg.snr_db) == (128, 64, 8, 25.5)


def test_config_from_env_rejects_unparsable_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """env 值无法解析时抛 :class:`ConfigError`（不静默回退默认值）。"""
    monkeypatch.setenv("ENV_SPARSEFORGE_N", "not-an-int")
    with pytest.raises(ConfigError, match="无法解析"):
        load_config()


def test_config_from_env_validates_overridden_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """env 覆盖后的值同样受校验（``m >= n`` 经env 注入也必须被拒）。"""
    monkeypatch.setenv("ENV_SPARSEFORGE_N", "32")
    monkeypatch.setenv("ENV_SPARSEFORGE_M", "64")
    with pytest.raises(ConfigError, match="m < n"):
        load_config()


def test_config_as_dict_roundtrip() -> None:
    """``as_dict()`` 覆盖全部 dataclass 字段（含新增的 ``elbow_eps``，供报告/序列化）。"""
    d = BenchmarkConfig().as_dict()
    assert d["n"] == 256
    assert d["m"] == 96
    assert d["k"] == 12
    assert isinstance(d["lam_grid"], tuple)
    # 新增 elbow_eps 字段（拐点 k 估计阈值），默认值 0.25
    assert d["elbow_eps"] == pytest.approx(0.25)
    # 逐个校验全部 14 个字段存在（不再硬编码脆弱长度）
    for key in (
        "n",
        "m",
        "k",
        "snr_db",
        "coherence",
        "n_seeds",
        "base_seed",
        "max_iter",
        "lam_grid",
        "ensemble_size",
        "refine_rounds",
        "hpo_trials",
        "tau",
        "elbow_eps",
    ):
        assert key in d, f"as_dict 缺字段 {key}"
    assert len(d) == 14


def test_config_from_env_with_base_preserves_other_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """给定 ``base`` 时，未被 env 覆盖的字段沿用 base 值。"""
    base = BenchmarkConfig(n=111, m=55, k=7)
    monkeypatch.setenv("ENV_SPARSEFORGE_K", "9")
    cfg = config_from_env(BenchmarkConfig, base)
    assert cfg.k == 9
    assert cfg.n == 111
    assert cfg.m == 55


# --------------------------------------------------------------------------- 数据类型


def test_recovery_result_to_dict_casts_types() -> None:
    """:class:`RecoveryResult.to_dict` 把 numpy 标量转成 Python 原生类型。"""
    res = RecoveryResult(
        x_hat=np.zeros(3),
        support=np.array([0]),
        residual_norm=1.0,
        nmse=2.0,
        support_f1=0.5,
        n_iter=3,
        converged=True,
        solver="x",
    )
    d = res.to_dict()
    for key in ("nmse", "support_f1", "residual_norm", "elapsed_sec"):
        assert type(d[key]) is float, f"{key} 应为 float"
    for key in ("n_iter",):
        assert type(d[key]) is int, f"{key} 应为 int"
    for key in ("converged",):
        assert type(d[key]) is bool, f"{key} 应为 bool"


def test_instance_score_and_properties() -> None:
    """:class:`Instance` 的 ``m``/``n``/``true_support``/``score` 语义正确。"""
    A = np.eye(4, 8)
    x = np.zeros(8)
    x[[1, 5]] = [2.0, -3.0]
    y = A @ x
    inst = Instance(
        name="t", A=A, y=y, x_true=x, k_true=2, kind="wellcond", snr_db=30.0, seed=0
    )
    assert inst.m == 4
    assert inst.n == 8
    assert np.array_equal(inst.true_support, np.array([1, 5]))
    # 完美恢复
    assert inst.score(x) == (0.0, 1.0)
    # 全零估计 ⇒ NMSE=1、F1=0
    nmse, f1 = inst.score(np.zeros(8))
    assert nmse == pytest.approx(1.0)
    assert f1 == 0.0


def test_instance_score_zero_true_convention() -> None:
    """``x_true`` 全零时 NMSE 约定为 0.0（不除零）。"""
    A = np.eye(3, 6)
    inst = Instance(
        name="z",
        A=A,
        y=np.zeros(3),
        x_true=np.zeros(6),
        k_true=0,
        kind="wellcond",
        snr_db=0.0,
        seed=0,
    )
    assert inst.score(np.zeros(6)) == (0.0, 1.0)


def test_convergence_error_is_numeric_layer() -> None:
    """:class:`ConvergenceError` 属 E300 层，可被基类捕获（分层正确）。"""
    with pytest.raises(SparseForgeError) as ei:
        raise ConvergenceError("迭代未收敛", n_iter=200)
    assert isinstance(ei.value, ConvergenceError)
    assert ei.value.code == "E300"
    assert ei.value.context["n_iter"] == 200


def test_backend_error_is_declared() -> None:
    """:class:`BackendError` 存在且带 E500 码（可选后端缺失时的契约）。"""
    exc = BackendError("sklearn 不可用", backend="sklearn")
    assert exc.code == "E500"
    assert isinstance(exc, SparseForgeError)
    assert "backend=sklearn" in str(exc)


def test_datagen_error_is_declared() -> None:
    """:class:`DataGenError` 存在且带 E100 码。"""
    exc = DataGenError("未知 DGP kind", kind="nope")
    assert exc.code == "E100"
    assert isinstance(exc, SparseForgeError)
    assert "kind=nope" in str(exc)
