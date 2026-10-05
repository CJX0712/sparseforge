"""Protocol 契约：求解器、信号生成器、评估器。

作者：晨星 · CJX0712
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from core.types import Instance, RecoveryResult

__all__ = ["Evaluator", "Preprocessor", "SignalGenerator", "Solver"]


@runtime_checkable
class Solver(Protocol):
    """稀疏恢复求解器契约。

    约定：
    - ``solve(A, y, k)`` 中 ``A`` 形状 ``(m, n)``，``y`` 形状 ``(m,)``，``k`` 为稀疏度或 ``None``
    - 返回的 ``RecoveryResult.x_hat`` 形状必须为 ``(n,)`` 且全有限
    - 求解器**不得**访问真值 ``x_true``（防泄漏硬约束，由单测强制）
    """

    name: str

    def solve(self, A: np.ndarray, y: np.ndarray, k: int | None) -> RecoveryResult: ...


@runtime_checkable
class SignalGenerator(Protocol):
    """压缩感知实例生成器契约。"""

    def __call__(self, n: int, m: int, k: int, snr_db: float, seed: int) -> Instance: ...


@runtime_checkable
class Evaluator(Protocol):
    """评估器契约：对一批结果打分并返回可聚合的度量。"""

    def evaluate(self, inst: Instance, x_hat: np.ndarray) -> dict[str, float]: ...


@runtime_checkable
class Preprocessor(Protocol):
    """预处理契约：输入 ``(A, y)``，输出变换后的 ``(A, y)`` 与还原句柄。"""

    def fit_transform(self, A: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...

    def inverse_transform(self, x: np.ndarray, handle: object) -> np.ndarray: ...
