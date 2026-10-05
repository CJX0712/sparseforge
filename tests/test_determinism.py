"""确定性测试：同 seed 必得同结果（逐位相同，非近似）。

作者：晨星 · CJX0712
"""

from __future__ import annotations

import numpy as np
import pytest

from core.seed import DEFAULT_SEED, get_seed, make_rng, set_all
from cs.solvers import omp
from data.generators import KINDS, make_instance

# --------------------------------------------------------------------------- 熵入口


def test_invariant_make_instance_is_bit_identical() -> None:
    """不变式 23：同 ``(kind, n, m, k, snr_db, coherence, seed)`` 两次生成逐位相同。

    用 ``np.array_equal``（**逐位**）而非 ``allclose``——``allclose`` 会放过
    1e-9 量级的漂移，而基准复现要求的是完全可重复。确定性自检（``cli check``）
    也正是按逐位一致来判定的。
    """
    a = make_instance("wellcond", m=48, n=128, k=5, snr_db=20.0, seed=42)
    b = make_instance("wellcond", m=48, n=128, k=5, snr_db=20.0, seed=42)
    assert np.array_equal(a.A, b.A)
    assert np.array_equal(a.y, b.y)
    assert np.array_equal(a.x_true, b.x_true)
    assert a.name == b.name
    assert a.seed == b.seed


def test_invariant_different_seed_gives_different_instance() -> None:
    """不变式 23b：不同 seed 必须产出**不同**实例（否则 seed 根本没生效）。"""
    a = make_instance("wellcond", m=48, n=128, k=5, snr_db=20.0, seed=42)
    b = make_instance("wellcond", m=48, n=128, k=5, snr_db=20.0, seed=43)
    assert not np.array_equal(a.A, b.A)
    assert not np.array_equal(a.y, b.y)
    assert not np.array_equal(a.x_true, b.x_true)


def test_invariant_kind_offset_mapping_is_stable() -> None:
    """不变式 24：``_kind_offset(kind, seed)`` 在同一进程内跨调用稳定。

    这是刻意**不用** ``hash()`` 的原因：CPython 对 ``str`` 的 hash 在进程启动
    时用随机盐初始化（PYTHONHASHSEED），跨进程会变。若用它做 offset，同一
    seed 在不同进程会生成不同实例，整个基准不可复现。

    本测试锁定三件事：
    1. 同一 ``(kind, seed)`` 多次调用返回同一 offset（跨调用稳定）
    2. 不同 kind 映射到不同 offset（无碰撞）
    3. offset 恒为非负整数，且随 kind 单调递增
    """
    from data.generators import _kind_offset

    first = [_kind_offset(kind, 7) for kind in KINDS]
    second = [_kind_offset(kind, 7) for kind in KINDS]
    assert first == second, "同进程内重复调用得到不同 offset"
    # 无碰撞
    assert len(set(first)) == len(first)
    # 非负整数且随 kind 顺序单调递增
    assert all(isinstance(v, int) and v >= 0 for v in first)
    assert first == sorted(first)
    # seed 线性叠加：offset(kind, seed+1) == offset(kind, seed) + 1
    assert _kind_offset("wellcond", 8) == _kind_offset("wellcond", 7) + 1


def test_invariant_kind_offset_drives_reproducibility_across_kinds() -> None:
    """不变式 24b：不同 kind 因offset 不同而拿到**独立**随机流。

    若两个 kind 共享 offset，它们的 ``A`` 会用同一随机流的前缀，不可复现地
    互相"撞车"。这条验证 6 个 kind 两两之间的实例都不同。
    """
    seen: dict[str, np.ndarray] = {}
    for kind in KINDS:
        inst = make_instance(kind, m=32, n=64, k=4, snr_db=20.0, seed=1)
        seen[kind] = inst.A
    names = list(seen)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            assert not np.array_equal(seen[a], seen[b]), f"{a} 与 {b} 的 A 完全相同"


def test_invariant_make_rng_sequences_identical_after_set_all() -> None:
    """不变式 25：``set_all(seed)`` 后连续两次 ``make_rng()`` 产生**相同**序列。

    ``make_rng(None)`` 返回由全局 seed 派生的默认流——它**不是**有状态的
    游标，而是每次新建 ``default_rng(base)``，所以两次调用必然相同。这正是
    "唯一熵入口"设计的意义：任何模块取到的随机流都由全局 seed 决定。
    """
    set_all(999)
    r1 = make_rng().standard_normal(5)
    r2 = make_rng().standard_normal(5)
    assert np.array_equal(r1, r2)


def test_invariant_make_rng_offset_substreams_are_distinct() -> None:
    """不变式 25b：不同 ``offset`` 派生**互不干扰**的子流。

    ``SeedSequence([base, offset])`` 保证不同 offset 的流不重叠——这是模块
    docstring 承诺的"不同 benchmark cell / seed 之间随机流不会互相污染"。
    """
    set_all(999)
    a = make_rng(offset=1).standard_normal(50)
    b = make_rng(offset=2).standard_normal(50)
    assert not np.array_equal(a, b)
    # 同一 offset 两次调用仍相同（子流也是确定的）
    assert np.array_equal(a, make_rng(offset=1).standard_normal(50))


def test_invariant_set_all_changes_subsequent_draws() -> None:
    """不变式 25c：``set_all`` 换seed 后 ``make_rng`` 产出不同序列。"""
    set_all(999)
    first = make_rng().standard_normal(5)
    set_all(1234)
    second = make_rng().standard_normal(5)
    assert not np.array_equal(first, second)
    # 回到原seed ⇒ 完全复现
    set_all(999)
    assert np.array_equal(make_rng().standard_normal(5), first)


def test_invariant_set_all_also_seeds_legacy_numpy() -> None:
    """``set_all`` 同时设定 legacy ``np.random.seed``（供 SciPy/sklearn 用）。

    只设 Generator 而不管 legacy 全局状态的实现，会让 sklearn 内部的随机性
    逃出确定性控制。
    """
    set_all(31337)
    legacy_first = np.random.rand(4)
    set_all(31337)
    assert np.array_equal(np.random.rand(4), legacy_first)


def test_default_seed_contract() -> None:
    """``DEFAULT_SEED`` 是约定的项目种子，且 ``get_seed`` 初始即为它。"""
    assert DEFAULT_SEED == 20261005
    # conftest 的 autouse fixture 已把全局 seed 复位
    assert get_seed() == DEFAULT_SEED


def test_set_all_none_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """``set_all(None)`` 优先读 ``ENV_SPARSEFORGE_SEED`` 环境变量。"""
    monkeypatch.setenv("ENV_SPARSEFORGE_SEED", "777")
    assert set_all(None) == 777
    assert get_seed() == 777
    monkeypatch.delenv("ENV_SPARSEFORGE_SEED")
    assert set_all(None) == DEFAULT_SEED


# --------------------------------------------------------------------------- 求解/流水线层


def test_invariant_solver_output_is_bit_identical() -> None:
    """不变式 26：同一实例上两次 ``omp`` 的输出逐位相同（求解器无隐藏随机性）。"""
    inst = make_instance("wellcond", m=48, n=128, k=6, snr_db=20.0, seed=11)
    from core.config import SolveConfig

    cfg = SolveConfig(k=6, max_iter=200)
    r1 = omp(inst.A, inst.y, 6, cfg)
    r2 = omp(inst.A, inst.y, 6, cfg)
    assert np.array_equal(r1.x_hat, r2.x_hat)
    assert r1.residual_norm == r2.residual_norm
    assert np.array_equal(r1.support, r2.support)


def test_invariant_pipeline_run_is_bit_identical() -> None:
    """不变式 27：两次 ``pipeline.run(methods=["omp"])`` 的 NMSE **完全相同**。

    这是 ``cli.py check`` 子命令在做的事（确定性自检），此处用最小 cell
    （1 个 regime × 3 seeds）复现它。两点Pipeline 实例、独立跑完整流程，
    比较逐行 NMSE。
    """
    from pipeline.benchmark import SparseForgePipeline

    cells = (("wellcond", 48, 6, 25.0, 0.0),)

    def run() -> list[tuple]:
        pipe = SparseForgePipeline(cells=cells, n_seeds=3)
        rows = pipe.run(methods=["omp"])
        return [(r.instance, r.seed, r.method, r.nmse, r.support_f1) for r in rows]

    first = run()
    second = run()
    assert len(first) == 3
    assert first == second, "两次 pipeline.run 的结果不一致"
    # 逐位（不是近似）相等
    assert [r[3] for r in first] == [r[3] for r in second]


def test_invariant_all_kinds_instances_are_reproducible() -> None:
    """不变式 27b：全部 6 种 DGP regime 都逐位可复现（不止wellcond）。"""
    for kind in KINDS:
        a = make_instance(kind, m=32, n=64, k=4, snr_db=20.0, coherence=0.4, seed=8)
        b = make_instance(kind, m=32, n=64, k=4, snr_db=20.0, coherence=0.4, seed=8)
        assert np.array_equal(a.A, b.A), kind
        assert np.array_equal(a.y, b.y), kind
        assert np.array_equal(a.x_true, b.x_true), kind


def test_set_all_rejects_bad_offset() -> None:
    """``make_rng`` 拒绝非整数 offset（bool 显式拒绝）。"""
    from core.errors import ConfigError

    with pytest.raises(ConfigError, match="offset 必须是整数"):
        make_rng(offset="x")  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="offset 必须是整数"):
        make_rng(offset=True)  # type: ignore[arg-type]
