"""cs.solvers — 稀疏恢复求解器（Tier-1 纯 numpy + Tier-0 sklearn 适配）。

Tier-1（纯 numpy，零下载可跑）
    omp      Orthogonal Matching Pursuit        (Pati, Rezaiifar & Krishnamurthy 1993)
    cosamp   CoSaMP                            (Needell & Tropp 2009)
    iht      Iterative Hard Thresholding        (Blumensath & Davies 2009)
    fista    FISTA / LASSO                     (Beck & Teboulle 2009)
    amp      AMP + Onsager 修正 + 阻尼         (Donoho, Maleki & Montanari 2009)

Tier-0（scikit-learn，可选后端，缺失时自动降级）
    sk_omp / sk_lasso / sk_lars

所有求解器遵守同一条硬契约：**solve() 绝不接触 x_true**。
作者：晨星 · CJX0712
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from core.config import SolveConfig
from core.errors import ConvergenceError, NumericsError
from core.types import RecoveryResult
from cs.operators import check_finite, debias_refit, soft_threshold, spectral_norm

__all__ = [
    "BASELINES",
    "FLAGSHIP",
    "SOLVERS",
    "amp",
    "available_sklearn",
    "cosamp",
    "fista",
    "iht",
    "omp",
    "sk_lars",
    "sk_lasso",
    "sk_omp",
]

_EPS = 1e-12


def _default_lam(A: np.ndarray, y: np.ndarray, cfg: SolveConfig) -> float:
    """默认正则化系数：``lam = lam_ratio * max|A^T y|``。

    **不能**再乘以 k：``max|A^T y|`` 本身已随 k 单调增（k 越大相关峰越高），
    再乘 k 会使 lam 远超信号幅度 ⇒ 软阈值把整向量压成全零（实测 NMSE=1.0、
    F1=0.0 的根因）。``lam_ratio`` 由 hpo 在同一 CV 口径上搜索。
    """
    if cfg.lam is not None:
        return float(cfg.lam)
    aty = float(np.max(np.abs(A.T @ y))) if y.size else 0.0
    return max(0.02 * aty, _EPS)


def _support_of(x: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    return np.flatnonzero(np.abs(x) > eps)


def _finish(
    x: np.ndarray,
    A: np.ndarray,
    y: np.ndarray,
    k: int | None,
    cfg: SolveConfig,
    name: str,
    t0: float,
    n_iter: int,
    converged: bool,
    debias: bool | None = None,
    meta: dict | None = None,
) -> RecoveryResult:
    """统一收尾：可选 debias、算残差/支撑集、计时。"""
    use_debias = cfg.debias if debias is None else debias
    if use_debias:
        sup = _support_of(x)
        if sup.size > 0:
            x = debias_refit(A, y, sup)
    check_finite(x)
    support = _support_of(x)
    r = y - A @ x
    return RecoveryResult(
        x_hat=x,
        support=support,
        residual_norm=float(np.linalg.norm(r)),
        nmse=float("nan"),
        support_f1=float("nan"),
        n_iter=int(n_iter),
        converged=bool(converged),
        solver=name,
        elapsed_sec=time.perf_counter() - t0,
        meta=dict(meta or {}),
    )


def _ls_on_support(A: np.ndarray, y: np.ndarray, sup: np.ndarray) -> np.ndarray:
    """在支撑集上最小二乘，返回**全尺寸**解向量。"""
    x = np.zeros(A.shape[1])
    if sup.size == 0:
        return x
    coef, *_ = np.linalg.lstsq(A[:, sup], y, rcond=None)
    x[sup] = coef
    return x


# --------------------------------------------------------------------------- OMP
def omp(
    A: np.ndarray, y: np.ndarray, k: int | None, cfg: SolveConfig | None = None
) -> RecoveryResult:
    """Orthogonal Matching Pursuit：每轮选最大相关列，在支撑集上最小二乘重拟合。"""
    cfg = cfg or SolveConfig()
    k = cfg.k if k is None else k
    if k is None:
        raise NumericsError("OMP 需要显式稀疏度 k")
    t0 = time.perf_counter()
    n = A.shape[1]
    k = int(min(max(1, k), n))
    sup: list[int] = []
    x = np.zeros(n)
    r = y.copy()
    for _ in range(k):
        corr = np.abs(A.T @ r)
        if sup:
            corr[np.asarray(sup)] = -np.inf
        i = int(np.argmax(corr))
        if not np.isfinite(corr[i]) or corr[i] <= 0:
            break
        sup.append(i)
        x = _ls_on_support(A, y, np.asarray(sup))
        r = y - A @ x
    return _finish(
        x,
        A,
        y,
        k,
        cfg,
        "omp",
        t0,
        len(sup),
        len(sup) == k,
        debias=False,
        meta={"k_used": len(sup)},
    )


# ----------------------------------------------------------------------- CoSaMP
def cosamp(
    A: np.ndarray, y: np.ndarray, k: int | None, cfg: SolveConfig | None = None
) -> RecoveryResult:
    """CoSaMP：每轮 top-2k 候选 + top-k 采样 + 最小二乘 + 残差回灌。"""
    cfg = cfg or SolveConfig()
    k = cfg.k if k is None else k
    if k is None:
        raise NumericsError("CoSaMP 需要显式稀疏度 k")
    t0 = time.perf_counter()
    n = A.shape[1]
    k = int(min(max(1, k), n // 2))
    sup = np.zeros(0, dtype=int)
    x = np.zeros(n)
    r = y.copy()
    for _ in range(cfg.max_iter):
        corr = np.abs(A.T @ r)
        cand = np.argpartition(corr, -min(2 * k, n))[-min(2 * k, n) :]
        pool = np.union1d(sup, cand)
        if pool.size == 0:
            break
        xs = _ls_on_support(A, y, pool)
        # top-k 必须按**幅值**取，不能用 _support_of（它按索引升序，切片取前 k 个）
        nz = np.flatnonzero(np.abs(xs) > 1e-10)
        if nz.size == 0:
            break
        top = nz[np.argsort(-np.abs(xs[nz]))][:k]
        x_new = _ls_on_support(A, y, top)
        r_new = y - A @ x_new
        stop = np.linalg.norm(r_new) >= np.linalg.norm(r) - 1e-12
        x, r, sup = x_new, r_new, top
        if stop:
            break
    return _finish(
        x,
        A,
        y,
        k,
        cfg,
        "cosamp",
        t0,
        cfg.max_iter,
        True,
        debias=False,
        meta={"k_used": int(sup.size)},
    )


# --------------------------------------------------------------------------- IHT
def iht(
    A: np.ndarray, y: np.ndarray, k: int | None, cfg: SolveConfig | None = None
) -> RecoveryResult:
    """IHT：硬阈值梯度迭代，步长 1/||A||_2^2。"""
    cfg = cfg or SolveConfig()
    k = cfg.k if k is None else k
    if k is None:
        raise NumericsError("IHT 需要显式稀疏度 k")
    t0 = time.perf_counter()
    n = A.shape[1]
    k = int(min(max(1, k), n))
    step = 1.0 / max(spectral_norm(A) ** 2, _EPS)
    x = np.zeros(n)
    prev = np.inf
    it = 0
    for _it in range(1, cfg.max_iter + 1):
        r = y - A @ x
        full = x + step * (A.T @ r)
        # 硬阈值：保留幅值最大的 k 个坐标（原值，不做软收缩）
        keep = np.argpartition(np.abs(full), -k)[-k:]
        keep = keep[np.argsort(-np.abs(full[keep]))]
        x_new = np.zeros(n)
        x_new[keep] = full[keep]
        nrm = float(np.linalg.norm(y - A @ x_new))
        x = x_new
        if abs(prev - nrm) < cfg.tol:
            break
        prev = nrm
    return _finish(
        x,
        A,
        y,
        k,
        cfg,
        "iht",
        t0,
        it,
        True,
        debias=True,
        meta={"k_used": int(_support_of(x).size)},
    )


# ------------------------------------------------------------------------- FISTA
def fista(
    A: np.ndarray, y: np.ndarray, k: int | None, cfg: SolveConfig | None = None
) -> RecoveryResult:
    """FISTA 求解 LASSO ``min 1/2||Ax-y||^2 + lam*||x||_1``（Beck & Teboulle 2009）。

    lam 由 ``cfg.lam`` 或 ``k`` 推导：``lam = lam_scale * k * max|A^T y|``。
    """
    cfg = cfg or SolveConfig()
    t0 = time.perf_counter()
    n = A.shape[1]
    step = 1.0 / max(spectral_norm(A) ** 2, _EPS)
    lam = _default_lam(A, y, cfg)

    def objective(z: np.ndarray) -> float:
        return 0.5 * float(np.sum((A @ z - y) ** 2)) + lam * float(np.sum(np.abs(z)))

    x = np.zeros(n)
    z = x.copy()
    t = 1.0
    prev_obj = objective(x)
    it = 0
    for _it in range(1, cfg.max_iter + 1):
        grad = A.T @ (A @ z - y)
        x_new = soft_threshold(z - step * grad, lam)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        z_new = x_new + ((t - 1.0) / t_new) * (x_new - x)
        obj = objective(x_new)
        if obj > prev_obj:  # 自适应重启：保证单调
            t_new = 1.0
            z_new = x_new
            obj = prev_obj
        x, z, t = x_new, z_new, t_new
        if abs(prev_obj - obj) <= cfg.tol * max(1.0, abs(prev_obj)):
            prev_obj = obj
            break
        prev_obj = obj
    return _finish(
        x,
        A,
        y,
        k,
        cfg,
        "fista",
        t0,
        it,
        True,
        debias=True,
        meta={"lam": lam, "objective": prev_obj},
    )


# --------------------------------------------------------------------------- AMP
def amp(
    A: np.ndarray, y: np.ndarray, k: int | None, cfg: SolveConfig | None = None
) -> RecoveryResult:
    """AMP（Bayes-optimal / LASSO 版）：近似消息传递 + Onsager 修正 + 阻尼 + 发散检测。

    迭代形式（Donoho, Maleki & Montanari 2009, Algorithm 1）：

    .. code-block:: text

        r = y - A x + A t             # (m,)  t 存 R^m，Onsager 修正回投到观测空间
        z = A^T r                     # (n,)
        x_new = eta_lambda(z)         # (n,)
        t = A η(A^T r) - A^T y         # (m,)  <- Onsager 项，定义在 R^m

    等价且更稳的写法（避免反复乘 A）：``t_new = x_new - eta(A^T y - A^T A x)`` 不成立
    （维度与符号都不对）。**实测踩坑**：把 Onsager 写成 ``x_new - eta(A^T r - A^T y)``
    并让 t 留在 R^n，会使 ``z = A^T r + t`` 在小 lam 下**必然发散**
    （lam<=0.3·max|A^T y| 时 100% ConvergenceError，lam>=0.6·max 恒全零解），
    根因是 ``-A^T y`` 这一项被重复计入。正确形式只保留 ``t = A·eta(A^T r) - A^T y``，
    它恰好是 ``eta`` 在 ``A^T y`` 处的线性化抵消。

    去掉 Onsager 项（``t`` 恒为 0）后 AMP 退化为 IHT，是消融 A1 的对照。
    """
    cfg = cfg or SolveConfig()
    t0 = time.perf_counter()
    m, n = A.shape
    lam = _default_lam(A, y, cfg)
    alpha = float(np.clip(cfg.damping, 0.0, 1.0))

    x = np.zeros(n)
    t = np.zeros(m)  # Onsager 修正回投到观测空间
    diverged = False
    it = 0
    ynorm = max(float(np.linalg.norm(y)), _EPS)
    for _it in range(1, cfg.max_iter + 1):
        r = y - A @ x + t  # (m,)  r = y - Ax + t
        x_new = soft_threshold(A.T @ r, lam)  # (n,)
        t_new = A @ (x_new - x)  # (m,)  Onsager: A·(eta(A^T r) - x)
        if alpha < 1.0:
            x_new = alpha * x_new + (1.0 - alpha) * x
        if not np.all(np.isfinite(x_new)) or float(np.linalg.norm(x_new)) > 50.0 * ynorm:
            diverged = True
            break
        delta = float(np.linalg.norm(x_new - x))
        x = x_new
        t = t_new
        # 收敛判据只看 x 的变化。**不能**把 t 的变化计入：Onsager 项每轮都在变，
        # 用 ||Δx||+||Δt|| 会使 delta 恒不衰减（实测 it=2 即误判收敛 / 恒不收敛）。
        if delta <= cfg.tol * max(1.0, float(np.linalg.norm(x))):
            break
    if diverged:
        raise ConvergenceError(
            "AMP 发散（已启用阻尼，可下调 damping 或改用 CoSaMP）", n_iter=it
        )
    return _finish(
        x,
        A,
        y,
        k,
        cfg,
        "amp",
        t0,
        it,
        not diverged,
        debias=True,
        meta={"lam": lam, "onsager": True, "damping": alpha, "n_iter_done": it},
    )


# ------------------------------------------------------------------ sklearn Tier-0
def available_sklearn() -> bool:
    try:
        import sklearn.linear_model  # noqa: F401
    except ImportError:  # pragma: no cover
        return False
    return True


def _sk_solve(
    name: str,
    estimator: object,
    A: np.ndarray,
    y: np.ndarray,
    k: int | None,
    cfg: SolveConfig,
    t0: float,
) -> RecoveryResult:
    try:
        est = estimator  # type: ignore[assignment]
        est.fit(A, y)  # type: ignore[attr-defined]
        x = np.asarray(est.coef_, dtype=np.float64).ravel()  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover - 后端 API 漂移兜底
        raise NumericsError("sklearn 后端失败", solver=name, err=str(exc)) from exc
    if x.shape[0] != A.shape[1]:  # pragma: no cover
        raise NumericsError("sklearn 后端输出维度不符", solver=name, got=int(x.shape[0]))
    return _finish(x, A, y, k, cfg, name, t0, 0, True, debias=True, meta={"backend": "sklearn"})


def sk_omp(
    A: np.ndarray, y: np.ndarray, k: int | None, cfg: SolveConfig | None = None
) -> RecoveryResult:
    """Tier-0：sklearn ``OrthogonalMatchingPursuit``。"""
    cfg = cfg or SolveConfig()
    t0 = time.perf_counter()
    from sklearn.linear_model import OrthogonalMatchingPursuit

    kk = cfg.k if k is None else k
    if kk is None:
        raise NumericsError("sk_omp 需要显式稀疏度 k")
    est = OrthogonalMatchingPursuit(n_nonzero_coefs=int(kk), fit_intercept=False)
    return _sk_solve("sk_omp", est, A, y, kk, cfg, t0)


def sk_lasso(
    A: np.ndarray, y: np.ndarray, k: int | None, cfg: SolveConfig | None = None
) -> RecoveryResult:
    """Tier-0：sklearn ``Lasso``（坐标下降）。``alpha`` 由 lam_grid 给出。"""
    cfg = cfg or SolveConfig()
    t0 = time.perf_counter()
    from sklearn.linear_model import Lasso

    kk = cfg.k if k is None else k
    kk = 1 if kk is None else int(kk)
    aty = float(np.max(np.abs(A.T @ y))) if y.size else 0.0
    alpha = float(cfg.lam) if cfg.lam is not None else max(0.02 * aty, _EPS)
    est = Lasso(
        alpha=alpha, fit_intercept=False, max_iter=max(2000, cfg.max_iter * 10), tol=1e-10
    )
    return _sk_solve("sk_lasso", est, A, y, kk, cfg, t0)


def sk_lars(
    A: np.ndarray, y: np.ndarray, k: int | None, cfg: SolveConfig | None = None
) -> RecoveryResult:
    """Tier-0：sklearn ``LassoLars``（LARS 路径）。"""
    cfg = cfg or SolveConfig()
    t0 = time.perf_counter()
    from sklearn.linear_model import LassoLars

    alpha = float(cfg.lam) if cfg.lam is not None else 1e-3
    est = LassoLars(alpha=alpha, fit_intercept=False, max_iter=500)
    return _sk_solve("sk_lars", est, A, y, k, cfg, t0)


#: 全部可用求解器（Tier-1 恒可用，Tier-0 缺失时抛 BackendError 由 pipeline 降级）
SOLVERS: dict[str, Callable[..., RecoveryResult]] = {
    "omp": omp,
    "cosamp": cosamp,
    "iht": iht,
    "fista": fista,
    "amp": amp,
    "sk_omp": sk_omp,
    "sk_lasso": sk_lasso,
    "sk_lars": sk_lars,
}


#: 强基线集合（用于门禁对比的"最强单法"）
#: 含重加权家族（Candes-Wakin-Boyd 2008）——**不给旗舰留"对手太弱"的后门**，
#: 否则重加权本身就是旗舰的一半，赢它等于自己打自己。
def _build_baselines() -> tuple[str, ...]:
    from cs.reweighted import amp_rl1, irl1_omp, reweighted_fista

    extra = {
        "rw_fista": reweighted_fista,
        "irl1_omp": irl1_omp,
        "amp_rl1": amp_rl1,
    }
    SOLVERS.update(extra)
    return (*SOLVERS.keys(),)


BASELINES: tuple[str, ...] = _build_baselines()

#: 旗舰名（实现见 cs.fusion）
FLAGSHIP = "cs_fuse"
