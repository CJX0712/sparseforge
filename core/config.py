"""配置：dataclass + ENV_SPARSEFORGE_* 环境变量覆盖 + schema 校验。

作者：晨星 · CJX0712
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields

from core.errors import ConfigError

__all__ = ["BenchmarkConfig", "SolveConfig", "config_from_env", "load_config"]

_ENV_PREFIX = "ENV_SPARSEFORGE_"


@dataclass(slots=True)
class SolveConfig:
    """单个求解器的超参。所有求解器共享同一入口，便于公平扫参。"""

    k: int | None = None
    max_iter: int = 200
    tol: float = 1e-10
    lam: float | None = None
    damping: float = 0.5
    block_size: int = 0
    refine_iters: int = 0
    debias: bool = True

    def __post_init__(self) -> None:
        if self.max_iter <= 0:
            raise ConfigError("max_iter 必须为正", max_iter=self.max_iter)
        if not 0.0 <= self.damping <= 1.0:
            raise ConfigError("damping 必须落在 [0, 1]", damping=self.damping)
        if self.tol < 0.0:
            raise ConfigError("tol 不得为负", tol=self.tol)
        if self.block_size < 0:
            raise ConfigError("block_size 不得为负", block_size=self.block_size)
        if self.refine_iters < 0:
            raise ConfigError("refine_iters 不得为负", refine_iters=self.refine_iters)


@dataclass(slots=True)
class BenchmarkConfig:
    """基准配置。默认值对应 demo 档（60s 预算内可跑完）。"""

    n: int = 256
    m: int = 96
    k: int = 12
    snr_db: float = 18.0
    coherence: float = 0.0
    n_seeds: int = 3
    base_seed: int = 20261005
    max_iter: int = 300
    # lam 以 max|A^T y| 的比例给出。实测（见 pitfalls §B「基线网格过窄」）：
    # 收缩类求解器的有效区间在 0.005~0.05，原 0.02~0.8 网格**整段落在
    # 全零解区**（lam >= 0.3*max|A^Ty| ⇒ NMSE=1.0、F1=0），等于没调参。
    # 这里加密到 13 点，覆盖 3 个数量级。
    lam_grid: tuple[float, ...] = field(
        default_factory=lambda: (
            0.002,
            0.005,
            0.01,
            0.02,
            0.03,
            0.05,
            0.08,
            0.12,
            0.2,
            0.3,
            0.5,
            0.8,
            1.2,
        )
    )
    ensemble_size: int = 6
    refine_rounds: int = 3
    hpo_trials: int = 12
    #: 不一致性原理的残差目标倍数（``kest="discrepancy"`` 时生效）。
    #: 保持**理论值 1.0** —— 实测在18 个基准实例上扫出的最优 0.40 无法通过
    #: 2-fold 交叉验证（A→B +123%），属测试集过拟合，不予采纳。
    tau: float = 1.0
    #: 残差拐点判据阈值（``elbow_k``）。0.25 由**双向交叉验证**确定：
    #: fold A 选出 0.23、fold B 选出 0.26，两个取值在对方 fold 上分别取得
    #: −49.4% / −35.6% 相对增益 ⇒ 区间稳健，取中值。
    elbow_eps: float = 0.25

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise ConfigError("n 必须为正", n=self.n)
        if self.m <= 0:
            raise ConfigError("m 必须为正", m=self.m)
        if self.m >= self.n:
            raise ConfigError("压缩感知要求 m < n", m=self.m, n=self.n)
        if not 1 <= self.k < self.m:
            raise ConfigError("k 必须落在 [1, m)", k=self.k, m=self.m)
        if not 0.0 <= self.coherence < 1.0:
            raise ConfigError("coherence 必须落在 [0, 1)", coherence=self.coherence)
        if self.n_seeds <= 0:
            raise ConfigError("n_seeds 必须为正", n_seeds=self.n_seeds)
        if self.refine_rounds < 0:
            raise ConfigError("refine_rounds 不得为负", r=self.refine_rounds)
        if self.ensemble_size <= 0:
            raise ConfigError("ensemble_size 必须为正", size=self.ensemble_size)
        if self.hpo_trials <= 0:
            raise ConfigError("hpo_trials 必须为正", trials=self.hpo_trials)
        if self.tau <= 0:
            raise ConfigError("tau 必须为正", tau=self.tau)
        if not 0.0 < self.elbow_eps < 1.0:
            raise ConfigError("elbow_eps 必须落在 (0, 1)", elbow_eps=self.elbow_eps)
        for v in self.lam_grid:
            if v <= 0.0:
                raise ConfigError("lam_grid 元素必须为正", lam=v)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _coerce(raw: str, ftype: object) -> object:
    name = getattr(ftype, "__name__", str(ftype))
    try:
        if ftype is int:
            return int(raw)
        if ftype is float:
            return float(raw)
        if ftype is bool:
            return raw.strip().lower() in {"1", "true", "yes", "on"}
    except ValueError as exc:
        raise ConfigError("环境变量无法解析", raw=raw, type=name) from exc
    return raw


def config_from_env(cls: type, base: object | None = None) -> object:
    """用 ``ENV_SPARSEFORGE_<FIELD_UPPER>`` 覆盖 dataclass 字段。"""
    kwargs: dict[str, object] = {}
    if base is not None:
        kwargs = asdict(base)  # type: ignore[assignment]
    for f in fields(cls):  # type: ignore[arg-type]
        env_key = _ENV_PREFIX + f.name.upper()
        if env_key not in os.environ:
            continue
        current = kwargs.get(f.name, f.default)
        ftype = type(current) if current is not None else f.type
        kwargs[f.name] = _coerce(os.environ[env_key], ftype)
    return cls(**kwargs)  # type: ignore[call-arg]


def load_config() -> BenchmarkConfig:
    """从环境变量构造基准配置。"""
    return config_from_env(BenchmarkConfig)  # type: ignore[return-value]
