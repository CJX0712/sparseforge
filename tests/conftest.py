"""共享测试夹具：确定性隔离、求解器参数化、小规模实例。

作者：晨星 · CJX0712
"""

from __future__ import annotations

import numpy as np
import pytest

from core.config import SolveConfig
from core.seed import DEFAULT_SEED, set_all
from cs.solvers import SOLVERS, available_sklearn


def pytest_configure(config: pytest.Config) -> None:
    """注册 ``slow`` 标记（子进程跑完整基准的用例）。

    这样 ``pytest -m "not slow"`` 可以跳过所有子进程冒烟用例。
    """
    config.addinivalue_line("markers", "slow: marks tests as slow (subprocess CLI runs)")


#: 纯 numpy（Tier-1）+ sklearn（Tier-0）求解器名。
#: ``amp`` / ``amp_rl1`` **故意不在此列**：实测在 wellcond 小实例上 100% 发散
#: （见 ``test_solvers.py::test_xfail_amp_diverges_on_wellcond``），把它们放进
#: 参数化夹具会让整个参数化网格失败。发散本身由独立 xfail 测试锁定。
ALL_SOLVER_NAMES: tuple[str, ...] = (
    "omp",
    "cosamp",
    "iht",
    "fista",
    "sk_omp",
    "sk_lasso",
    "sk_lars",
    "rw_fista",
    "irl1_omp",
)

SKLEARN_SOLVER_NAMES: tuple[str, ...] = ("sk_omp", "sk_lasso", "sk_lars")


@pytest.fixture(autouse=True)
def _isolate_global_seed() -> None:
    """每个测试前把全局熵源复位到 ``DEFAULT_SEED``。

    ``core.seed.set_all`` 是全项目唯一熵入口且写全局状态；不隔离的话测试间
    会互相污染，导致确定性测试偶发失败。
    """
    set_all(DEFAULT_SEED)


@pytest.fixture
def rng() -> np.random.Generator:
    """独立随机流（不依赖全局 seed 状态）。"""
    return np.random.default_rng(20261005)


@pytest.fixture
def solver_fn(request: pytest.FixtureRequest):
    """参数化求解器：``solver_fn(A, y, k, cfg) -> RecoveryResult``。

    用法::

        @pytest.mark.parametrize("solver_fn", ALL_SOLVER_NAMES, indirect=True)
        def test_x(solver_fn): ...

    sklearn 不可用时自动跳过对应名字，而不是让测试崩在``ImportError`` 上。
    """
    name = request.param
    if name in SKLEARN_SOLVER_NAMES and not available_sklearn():
        pytest.skip(f"sklearn 不可用，跳过 {name}")
    return SOLVERS[name]


@pytest.fixture
def tiny_instance():
    """小规模压缩感知实例：``m=24, n=64, k=3``，高 SNR 的良态 ``wellcond``。

    规模刻意压到最小以保证全量测试秒级完成，同时 ``m/n = 0.375`` 远低于
    经典相变阈值，OMP 之类的贪心法在该实例上可稳定工作。
    """
    from data.generators import make_instance

    return make_instance(kind="wellcond", m=24, n=64, k=3, snr_db=25.0, seed=5)


@pytest.fixture
def tiny_cfg() -> SolveConfig:
    """与 :func:`tiny_instance` 配套的求解配置。"""
    return SolveConfig(k=3, max_iter=200, debias=True)


@pytest.fixture
def noiseless_problem():
    """无噪声、列归一化、列归一化随机测量矩阵 ``(A, x, y=A x, support)``。

    ``m=64, n=256, k=4``。无噪声是"精确恢复"不变量的**唯一**合法测试床：
    带噪声时 NMSE 的下界是噪声能量比（40dB ⇒≈1e-5），不可能达到 1e-10。
    """
    gen = np.random.default_rng(700)
    m, n, k = 64, 256, 4
    A = gen.standard_normal((m, n))
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    support = np.sort(gen.choice(n, size=k, replace=False))
    x = np.zeros(n)
    x[support] = gen.standard_normal(k)
    return A, x, A @ x, support
