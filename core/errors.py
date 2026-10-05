"""异常层级：SparseForgeError 基类 + E100~E500 五个语义层。

作者：晨星 · CJX0712
"""

from __future__ import annotations

__all__ = [
    "BackendError",
    "ConfigError",
    "ConvergenceError",
    "DataGenError",
    "NumericsError",
    "SparseForgeError",
]


class SparseForgeError(Exception):
    """所有 SparseForge 异常的基类，带稳定错误码。"""

    code = "E000"

    def __init__(self, message: str, **context: object) -> None:
        self.context = context
        detail = "".join(f" | {k}={v}" for k, v in context.items())
        super().__init__(f"[{self.code}] {message}{detail}")


class DataGenError(SparseForgeError):
    """E100：合成数据 / 载入数据不满足契约（如 m >= n、k 越界、SNR 非法）。"""

    code = "E100"


class NumericsError(SparseForgeError):
    """E200：数值不变量被破坏（Gram 不对称、非有限值、秩亏、支撑集非法）。"""

    code = "E200"


class ConvergenceError(SparseForgeError):
    """E300：迭代在最大迭代数内未收敛（ISTA/FISTA/AMP 发散或停滞）。"""

    code = "E300"


class ConfigError(SparseForgeError):
    """E400：配置或环境变量覆盖非法。"""

    code = "E400"


class BackendError(SparseForgeError):
    """E500：可选后端不可用或 API 漂移。"""

    code = "E500"
