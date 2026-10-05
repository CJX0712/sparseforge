"""data.generators — 压缩感知实例生成。

覆盖 6 种 regime，用于避免"单一 regime 一家独大"的病态基准：
    wellcond    高斯随机测量，SNR 高（易）
    coherent    高斯测量 + 秩一尖峰（相干硬实例，OMP 类退化）
    low_snr     高斯测量，SNR 很低（噪声主导）
    clustered   分块稀疏（group-sparse）信号，标量 ℓ1 模型失配
    fourier     部分傅里叶测量（结构化算子）
    heavy_tail  尖峰厚尾幅度（outlier 主导能量）

每个生成器固定 seed ⇒ 可复现。**真值仅用于打分，绝不传给任何求解器。**

作者：晨星 · CJX0712
"""

from __future__ import annotations

import numpy as np

from core.errors import DataGenError
from core.seed import make_rng
from core.types import Instance

__all__ = [
    "KINDS",
    "make_fourier_matrix",
    "make_gaussian_matrix",
    "make_instance",
    "make_spiked_matrix",
]

KINDS = ("wellcond", "coherent", "low_snr", "clustered", "fourier", "heavy_tail")

_EPS = 1e-12


def _normalize_columns(A: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(A, axis=0)
    norms = np.where(norms < _EPS, 1.0, norms)
    return A / norms


def make_gaussian_matrix(m: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """iid 高斯测量矩阵，列归一化。AMP 理论要求 iid Gaussian，此处保持。"""
    return _normalize_columns(rng.standard_normal((m, n)) / np.sqrt(m))


def make_fourier_matrix(m: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """实值部分傅里叶测量矩阵：堆叠 ``cos`` 与 ``sin`` 两半，列归一化。

    不能直接取 ``Re(exp(i*theta))``：``cos`` 是偶函数，频率 ``+r`` 与 ``-r``
    的实部**完全相同** ⇒ 产生严格重复列（实测相干度 1.0000、条件数 6.5e13），
    使 OMP/IHT 全部崩溃。堆叠 cos+sin 是标准的实值化方式，既保持部分傅里叶
    的 RIP 结构又消除重复列。
    """
    half = max(1, m // 2)
    freqs = rng.choice(n, size=half, replace=False)
    cols = np.arange(n)
    theta = 2.0 * np.pi * np.outer(freqs, cols) / n
    A = np.vstack([np.cos(theta), np.sin(theta)[: m - half]])
    return _normalize_columns(A / np.sqrt(2.0 * half))


def make_spiked_matrix(
    m: int, n: int, rng: np.random.Generator, spike: float = 0.97
) -> np.ndarray:
    """高斯测量 + 秩一尖峰：制造一对近乎共线的列（相干硬实例）。

    做法：取两个随机列索引，把第 j 列替换为 ``spike * col_i + sqrt(1-spike^2) * col_j``
    的归一化版本。

    **实测踩坑（D-a，已修复）**：原实现写成 ``(1-s)*col_i + s*col_j``，
    这让 ``s->1`` 时新列≈原 col_j（相关性≈0），``s->0.5`` 时才最共线 ——
    **映射完全反向**。实测 ``coherence=0.9`` 的 coherent regime 相干度
    0.4363，竟低于 coherence=0.0 的 wellcond 0.4430，参数形同虚设。
    正确写法让两列夹角余弦 = spike（列已归一化且原两列近似正交）：
    ``<a_i, (s*a_i + sqrt(1-s^2)*a_j)/||.||> ≈ s``。

    Parameters
    ----------
    spike:
        目标相干度，落``(0, 1)``。越大越接近 ``coherence -> 1``。
    """
    if not 0.0 < spike < 1.0:
        raise DataGenError("spike 必须落在 (0, 1)", spike=spike)
    A = make_gaussian_matrix(m, n, rng)
    i, j = rng.choice(n, size=2, replace=False)
    perp = float(np.sqrt(max(1.0 - spike * spike, 0.0)))
    v = spike * A[:, i] + perp * A[:, j]
    nv = np.linalg.norm(v)
    if nv > _EPS:
        A[:, j] = v / nv
    return A


def _make_signal(
    n: int, k: int, rng: np.random.Generator, kind: str
) -> tuple[np.ndarray, int, dict[str, object]]:
    """返回 ``(x, n_groups, meta)``。"""
    if kind == "clustered":
        block = 8
        n_blocks = n // block
        n_active = max(1, min(n_blocks, round(k / block)))
        active = rng.choice(n_blocks, size=n_active, replace=False)
        x = np.zeros(n)
        groups = []
        for b in np.sort(active):
            lo = int(b) * block
            hi = min(n, lo + block)
            vals = rng.standard_normal(hi - lo)
            x[lo:hi] = vals
            groups.append((lo, hi))
        return x, n_active, {"block": block, "n_groups": n_active, "grouped": True}

    support = np.sort(rng.choice(n, size=k, replace=False))
    if kind == "heavy_tail":
        amp = rng.standard_t(df=3, size=k)
        amp = amp / max(np.max(np.abs(amp)), _EPS)
    else:
        amp = rng.standard_normal(k)
    x = np.zeros(n)
    x[support] = amp
    return x, k, {"grouped": False, "support": support.tolist()}


def make_instance(
    kind: str,
    n: int = 256,
    m: int = 96,
    k: int = 12,
    snr_db: float = 18.0,
    coherence: float = 0.0,
    seed: int = 0,
) -> Instance:
    """生成单个压缩感知实例。

    Parameters
    ----------
    kind:
        ``KINDS`` 之一。
    n, m, k:
        信号长度 / 测量数 / 稀疏度。要求 ``1 <= m < n`` 且 ``1 <= k < m``。
    snr_db:
        信噪比（dB），定义为 ``20*log10(||A x|| / ||noise||)``。
    coherence:
        ``coherent``  regime 下的尖峰程度，映射到 spike 值。
    seed:
        整数种子。同一 ``(kind, n, m, k, snr_db, coherence, seed)`` 必得同一实例。
    """
    if kind not in KINDS:
        raise DataGenError("未知 DGP kind", kind=kind, allowed=list(KINDS))
    if n <= 0 or m <= 0:
        raise DataGenError("n 与 m 必须为正", n=n, m=m)
    if m >= n:
        raise DataGenError("压缩感知要求 m < n", m=m, n=n)
    if not 1 <= k < m:
        raise DataGenError("k 必须落在 [1, m)", k=k, m=m)

    rng = make_rng(offset=_kind_offset(kind, seed))
    eff_snr = snr_db
    if kind == "low_snr":
        eff_snr = min(snr_db, 6.0)

    if kind == "fourier":
        A = make_fourier_matrix(m, n, rng)
    elif kind == "coherent":
        spike = 0.90 + 0.08 * float(np.clip(coherence, 0.0, 1.0))
        A = make_spiked_matrix(m, n, rng, spike=spike)
    else:
        A = make_gaussian_matrix(m, n, rng)

    x, n_groups, sig_meta = _make_signal(n, k, rng, kind)

    clean = A @ x
    sig_norm = float(np.linalg.norm(clean))
    noise_norm = sig_norm / (10.0 ** (eff_snr / 20.0))
    sigma = noise_norm / np.sqrt(m)
    noise = rng.standard_normal(m) * sigma
    y = clean + noise

    meta: dict[str, object] = {
        "kind": kind,
        "n_groups": n_groups,
        "sigma": float(sigma),
        "noise_norm": float(noise_norm),
        "clean_norm": sig_norm,
        **sig_meta,
    }
    return Instance(
        name=f"{kind}_snr{eff_snr:g}",
        A=A,
        y=y,
        x_true=x,
        k_true=int(k),
        kind=kind,
        snr_db=float(eff_snr),
        seed=int(seed),
        meta=meta,
    )


def _kind_offset(kind: str, seed: int) -> int:
    """把 (kind, seed) 稳定映射为一个非负 offset，避免 hash 随机化。"""
    table = {name: i + 1 for i, name in enumerate(KINDS)}
    return (table[kind] - 1) * 1_000_003 + int(seed)
