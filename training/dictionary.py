"""training.dictionary — MOD / K-SVD 字典学习（与稀疏恢复解耦的训练层）。

用途：在"信号并非相对给定字典 k-sparse"的场景下学出字典，再交 cs 层恢复。
本仓 benchmark 主线仍走"已知字典"路径（经典压缩感知设定），字典学习作为
独立可验证能力提供，并由单测锁定其两条核心不变量。

作者：晨星 · CJX0712
"""

from __future__ import annotations

import numpy as np

from core.errors import NumericsError

__all__ = ["ksvd_dictionary", "mod_dictionary", "synthesize_block_dictionary"]

_EPS = 1e-12


def synthesize_block_dictionary(
    n: int, n_atoms: int, rng: np.random.Generator, block: int = 4, decay: float = 0.85
) -> np.ndarray:
    """合成具有块状相关结构的字典（列归一化，模拟真实信号的组内相关性）。"""
    n_blocks = max(1, n // block)
    base = rng.standard_normal((n, n_blocks))
    idx = np.arange(n_atoms) % n_blocks
    coef = decay ** np.arange(block)[None, :]
    cols = base[:, idx] * coef
    norms = np.linalg.norm(cols, axis=0, keepdims=True)
    return cols / np.maximum(norms, _EPS)


def mod_dictionary(
    X: np.ndarray, n_atoms: int, n_iter: int = 30, init: np.ndarray | None = None
) -> np.ndarray:
    """MOD（Method of Directions）字典学习。

    固定字典 D，用 OMP 求稀疏系数后交替更新 D。返回列归一化字典。
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if X.shape[1] == 0:
        raise NumericsError("训练样本为空")
    n_atoms = int(min(n_atoms, X.shape[1] * 4))
    rng = np.random.default_rng(0)
    D = (
        np.asarray(init, dtype=np.float64)
        if init is not None
        else rng.standard_normal((n, n_atoms))
    )
    D = D / np.maximum(np.linalg.norm(D, axis=0, keepdims=True), _EPS)
    for _ in range(max(1, n_iter)):
        S = np.zeros((n_atoms, X.shape[1]))
        for j in range(X.shape[1]):
            xj = X[:, j]
            for _k in range(3):  # 3 轮 OMP 足够（合成数据块结构强）
                corr = np.abs(D.T @ xj)
                i = int(np.argmax(corr))
                if corr[i] <= _EPS:
                    break
                D_i = D[:, [i]]
                if j == 0:
                    pass
                c, *_ = np.linalg.lstsq(D_i, xj, rcond=None)
                resid = xj - D_i @ c
                xj = resid
            idx = np.argsort(-np.abs(D.T @ X[:, j]))[: max(1, n_atoms // 8)]
            c, *_ = np.linalg.lstsq(D[:, idx], X[:, j], rcond=None)
            S[idx, j] = c
        for i in range(n_atoms):
            rows = S[i] != 0
            if rows.sum() == 0:
                continue
            D[:, i] = (X[:, rows] @ S[i, rows]) / max(float(S[i, rows] @ S[i, rows]), _EPS)
        D = D / np.maximum(np.linalg.norm(D, axis=0, keepdims=True), _EPS)
    return D


def ksvd_dictionary(
    X: np.ndarray, n_atoms: int, n_iter: int = 20, block: int = 1
) -> np.ndarray:
    """K-SVD 字典学习：每轮对每个原子做近邻块 SVD 更新。

    ``block`` 控制 SVD 的近邻块大小（block=1 即经典 K-SVD）。
    """
    X = np.asarray(X, dtype=np.float64)
    n, m = X.shape
    n_atoms = int(min(n_atoms, m))
    rng = np.random.default_rng(0)
    D = rng.standard_normal((n, n_atoms))
    D = D / np.maximum(np.linalg.norm(D, axis=0, keepdims=True), _EPS)
    for _ in range(max(1, n_iter)):
        for j in range(m):
            xj = X[:, j]
            corr = D.T @ xj
            nb = np.argsort(-np.abs(corr))[: min(block, n_atoms)]
            Dn = D[:, nb]
            Q, _ = np.linalg.qr(Dn)
            r = xj - Q @ (Q.T @ xj)
            u, s, _vt = np.linalg.svd(np.column_stack([Dn, r]), full_matrices=False)
            take = min(len(nb) + 1, u.shape[1])
            D[:, nb] = u[:, : len(nb)] * s[: len(nb)]
            if take > len(nb) and u.shape[1] > len(nb):
                extra = D.shape[1] - len(nb)
                if extra > 0:
                    D[:, -extra:] = u[:, len(nb) : len(nb) + extra]
        D = D / np.maximum(np.linalg.norm(D, axis=0, keepdims=True), _EPS)
    return D
