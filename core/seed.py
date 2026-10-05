"""全局确定性熵入口：唯一 seed 入口，禁止任何模块另起 RandomState。

作者：晨星 · CJX0712
"""

from __future__ import annotations

import os
import random

import numpy as np

from core.errors import ConfigError

__all__ = ["DEFAULT_SEED", "get_seed", "make_rng", "set_all"]

DEFAULT_SEED = 20261005

_STATE: dict[str, int] = {"seed": DEFAULT_SEED}


def set_all(seed: int | None = None) -> int:
    """设定全局熵源。返回实际生效的 seed。

    这是全项目**唯一**允许设置随机性的入口。numpy 侧同时设定 legacy
    ``np.random.seed``（供 SciPy/sklearn 内部使用）与默认 Generator 的派生种子。
    """
    if seed is None:
        env = os.environ.get("ENV_SPARSEFORGE_SEED")
        seed = int(env) if env else DEFAULT_SEED
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ConfigError("seed 必须是整数", seed=repr(seed))
    seed = int(seed)
    if seed < 0 or seed >= 2**63:
        raise ConfigError("seed 必须落在 [0, 2**63)", seed=seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    _STATE["seed"] = seed
    return seed


def get_seed() -> int:
    """返回当前生效的全局 seed（从未调用 set_all 时返回 DEFAULT_SEED）。"""
    return _STATE["seed"]


def make_rng(offset: int | None = None) -> np.random.Generator:
    """从全局 seed 派生一个独立 Generator。

    ``offset=None`` 时返回由全局 seed 派生的默认流；给定 offset 时返回
    ``SeedSequence([global_seed, offset])`` 派生的**互不干扰**的子流。
    这一保证不同 benchmark cell / seed 之间的随机流不会互相污染。
    """
    base = get_seed()
    if offset is None:
        return np.random.default_rng(base)
    if isinstance(offset, bool) or not isinstance(offset, (int, np.integer)):
        raise ConfigError("offset 必须是整数", offset=repr(offset))
    return np.random.default_rng(np.random.SeedSequence([base, int(offset)]))
