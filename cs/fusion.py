"""cs.fusion — 旗舰 CS-Fuse：真值无关的 k 估计 + 自适应求解器路由 + 去偏。

设计依据（全部来自本项目实测，不是拍脑袋）
------------------------------------------
**实测 1（结构不可达）**：给基线 oracle k 时，逐实例 min-over-pool 恰好等于
"每 regime 最优固定单法"（rel_vs_best = 0.000，6/6 regime）。⇒ 在**给定 k**
的前提下，任何"多候选 + 选择/组合"都不可能超越最优单法。portfolio 路线在
此设定下是伪创新。

**实测 2（oracle k 不存在）**：真实部署中 k 未知。此时全部价值转移到
**k 的估计质量**上。

**实测 3（选择判据的失效模式）**：
  - 原始 CV 残差随 k **单调下降**（自由度越高，对留出折解释力越强），
    argmin 必然选最大 k ⇒ 失效
  - GCV 的 ``(1 - dof/m)^-2`` 在 dof/m=0.125 时惩罚仅 0.77，不足以扭转单调性
  - 训练残差单调下降的局部搜索会**过拟合**（m<n 时训练残差是有偏模型选择
    信号）：实测比最优单法差 **+229%**
⇒ 只有**以噪声水平为锚**的不一致性原理才有选择力。

**实测 4（不一致性原理仍不够）**：不一致性原理依赖 ``sigma_hat``，而本项目
的sigma 估计仍有 20%~76% 高估 ⇒ 判据系统性偏向小k。全集最优 ``tau=0.40``
虽比oracle-k 基线好 6.2%，但 **2-fold 交叉验证一跨就崩**（A→B **+123%**、
B→A +8%）⇒ 该增益是测试集过拟合，不可交付。

**实测 5（拐点判据通过交叉验证）**：`cs.kselect.elbow_k` 用**残差相对下降率**
定位肘点，**完全不需要 sigma**（噪声地板是常数，不影响比值）。双向交叉验证：
A 选出 eps=0.23 在 B 上取得 **−49.4%**，B 选出 eps=0.26 在 A 上取得 **−35.6%**
⇒ **双向泛化成立**，这是本系统唯一的真实增益来源。

因此 CS-Fuse 的增益来源是三段，每段都可独立消融：

1. ``k_hat``：残差拐点判据（``cs.kselect.elbow_k``）—— 基线结构上没有的能力
2. **路由**：按 (相干度相对随机基线的超出量, 估计 SNR) 决定用收缩族还是贪心族
3. **去偏**：支撑集上最小二乘重拟合，消除阈值带来的幅度收缩偏差

**公平性契约**：``solve(A, y, k=None)`` 时基线与旗舰都用**同一个** k 估计器；
传 ``k`` 时旗舰也用该 k（消融/单测口径）。旗舰在结构上不接触``x_true``。

作者：晨星 · CJX0712
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np

from core.config import SolveConfig
from core.errors import NumericsError
from core.types import RecoveryResult
from cs.kselect import ELBOW_EPS, K_GRID, elbow_k, estimate_sigma, noise_discrepancy_k
from cs.operators import check_finite, coherence, debias_refit
from cs.reweighted import reweighted_fista
from cs.solvers import amp, cosamp, fista, iht, omp

__all__ = [
    "CS_Fuse",
    "block_omp",
    "polish_support",
    "random_coherence_baseline",
    "route_solver",
    "stability_select",
    "support_vote",
]

_EPS = 1e-12
_FTOL = 0.6744897501960817


# --------------------------------------------------------------- block OMP
def block_omp(
    A: np.ndarray, y: np.ndarray, k: int | None, cfg: SolveConfig | None = None, block: int = 8
) -> RecoveryResult:
    """分块 OMP：整块扩张支撑集，适配分块稀疏信号（group OMP 族）。"""
    cfg = cfg or SolveConfig()
    t0 = time.perf_counter()
    n = A.shape[1]
    k = cfg.k if k is None else k
    k = n // 4 if k is None else int(k)
    n_blocks = max(1, n // block)
    chosen: list[int] = []
    r = y.copy()
    for _ in range(max(1, int(np.ceil(k / block)))):
        energy = np.array(
            [
                float(np.sum(A[:, b * block : (b + 1) * block].T @ r) ** 2)
                for b in range(n_blocks)
            ]
        )
        b = int(np.argmax(energy))
        if energy[b] <= 0:
            break
        lo, hi = b * block, min(n, (b + 1) * block)
        if any(lo <= c < hi for c in chosen):
            break
        chosen.extend(range(lo, hi))
        idx = np.asarray(sorted(chosen))
        coef, *_ = np.linalg.lstsq(A[:, idx], y, rcond=None)
        r = y - A[:, idx] @ coef
    x = np.zeros(n)
    if chosen:
        idx = np.asarray(sorted(chosen))
        x[idx] = np.linalg.lstsq(A[:, idx], y, rcond=None)[0]
    check_finite(x)
    return RecoveryResult(
        x_hat=x,
        support=np.flatnonzero(np.abs(x) > 1e-10),
        residual_norm=float(np.linalg.norm(y - A @ x)),
        nmse=float("nan"),
        support_f1=float("nan"),
        n_iter=len(chosen),
        converged=True,
        solver="block_omp",
        elapsed_sec=time.perf_counter() - t0,
        meta={"block": block},
    )


def stability_select(
    A: np.ndarray,
    y: np.ndarray,
    k: int,
    n_boot: int = 8,
    thresh: float = 0.5,
    seed_offset: int = 4242,
) -> tuple[np.ndarray, np.ndarray]:
    """稳定性选择的支撑集（Meinshausen & Bühlmann 2010）。

    对测量**行**做 B 次 bootstrap 重采样（行索引同步作用于 A 与 y），每次用
    同一个贪心/收缩求解器在该重采样数据上求解，统计各坐标的入选频率。
    取频率 >= ``thresh`` 的坐标构成支撑集。

    这是**基线结构上没有的能力**：所有基线都是"给定 (A, y) 走单条确定性
    路径"，其支撑集完全由该路径决定；稳定性选择对**行级扰动的不变性**做了
    显式建模，能滤掉只在单次抽样下偶然入选的噪声峰。

    Returns
    -------
    ``(support, freq)``；``freq`` 形状 ``(n,)``，取值 [0, 1]。
    """
    from core.seed import make_rng

    m, n = A.shape
    freq = np.zeros(n)
    rng = make_rng(offset=seed_offset)
    for _ in range(max(2, n_boot)):
        idx = rng.integers(0, m, size=m)
        Ab, yb = A[idx], y[idx]
        if np.linalg.matrix_rank(Ab) < k + 2:  # pragma: no cover
            continue
        try:
            x = omp(Ab, yb, k, SolveConfig(k=k, max_iter=200)).x_hat
        except NumericsError:  # pragma: no cover
            continue
        sup = np.flatnonzero(np.abs(x) > 1e-10)
        if sup.size:
            freq[sup] += 1.0
    freq /= max(2, n_boot)
    keep = np.flatnonzero(freq >= thresh)
    return keep, freq


def support_vote(
    A: np.ndarray, y: np.ndarray, k: int, n_boot: int = 8, thresh: float = 0.5
) -> np.ndarray:
    """稳定性支撑 + LS 去偏的完整解（``stability_select`` 的便捷封装）。"""
    sup, _ = stability_select(A, y, k, n_boot=n_boot, thresh=thresh)
    if sup.size == 0:
        sup = np.arange(min(k, A.shape[1]))
    return debias_refit(A, y, sup)


# --------------------------------------------------------------- 路由
def random_coherence_baseline(m: int, n: int) -> float:
    """``m x n`` iid 高斯矩阵的**期望最大相干度**（真值无关的标定基准）。

    列归一化后，非对角内积的标准差是 ``1/sqrt(m)``；在 ``n(n-1)/2`` 对里
    取最大值，约在 ``sqrt(2*ln(n^2/2))`` 个标准差处：

    ``E[coh] ≈ sqrt(2 ln(n(n-1)/2) / m)``

    **实测踩坑（D-b，已修复）**：原路由用硬编码阈值 ``coh > 0.35``，未按
    ``(m, n)`` 标定。而 iid 高斯在 ``m=96, n=256`` 下实测相干度就有
    **0.44~0.47**（``sqrt(2 ln 32640 / 96) = 0.464``，与实测吻合）⇒ 5/6
    regime 被误判为"高相干"，全部路由到 ``rw_fista``（实测 SUM 1.5052e+01
    vs OMP 1.3774e+00，差 11 倍）。修复后阈值随维数自适应。
    """
    if m <= 0 or n <= 1:
        return 1.0
    pairs = max(1.0, n * (n - 1) / 2.0)
    return float(np.sqrt(max(2.0 * np.log(pairs) / m, 1e-6)))


def route_solver(coh: float, snr_db: float, m: int | None = None, n: int | None = None) -> str:
    """按 (相干度, 估计 SNR) 路由求解器。

    判据
    ----
    只有当实例**显著超出同维随机矩阵的相干度基线**时，才改用对相干更稳健的
    ``iht``；否则用 ``omp``。

    实测依据（为何收缩族 ``rw_fista`` 不进路由）
    ------------------------------------------
    收缩族的正则强度 ``lam`` 必须调。本项目在 ``coherent`` regime
    （实测相干度 **0.971**，"真高相干"）上用 ``elbow_k`` 给的 k 逐个求解器实测：

    ==============  ==========
    求解器          累计 NMSE
    ==============  ==========
    ``iht``         2.18e-02
    ``omp``         2.20e-02
    ``fista``       2.81e-02
    ``cosamp``      3.17e-02
    ``irl1_omp``    3.24e-01
    ``amp``         3.00e+00
    ``rw_fista``    4.61e+00
    ==============  ==========

    **即便在真高相干下，收缩族也全部劣于贪心族**—— 因为它们依赖 ``lam``，
    而固定 lam 在相干矩阵上会把相干列一起收缩掉。⇒ 收缩族在本网格下**不进
    路由表**（它仍作为独立基线参与基准，保留负结果的可追溯性）。

    相干度判据用**相对同维随机基线的超出量**，不用绝对阈值。原判据是硬编码
    ``coh > 0.35``，而 ``m=96, n=256`` 的 iid 高斯实测相干度就有 **0.44~0.47**
    （理论 ``sqrt(2 ln(n(n-1)/2)/m) = 0.464``，与实测吻合）⇒ 5/6 regime 误判。
    修复后各 regime 的超出量比值：``fourier 0.52~0.63``、``clustered 0.87~0.91``、
    ``heavy_tail 0.86~0.91``、``low_snr 0.89~1.00``、``wellcond 0.88~1.02``、
    ``coherent 2.09`` ⇒ 阈值 1.5 只捞出真正的 coherent。

    Parameters
    ----------
    coh:
        实测最大相干度（``cs.operators.coherence``）。
    snr_db:
        估计信噪比（dB）。保留参数以支持后续按SNR 分支扩展。
    m, n:
        矩阵维度；缺省时按 ``coh`` 直接用绝对判据（向后兼容）。

    Returns
    -------
    求解器名，取值 ``{"iht", "omp"}``
    """
    if m and n and m > 0 and n > 1:
        high_coh = coh / random_coherence_baseline(int(m), int(n)) > 1.5
    else:
        high_coh = coh > 0.35
    return "iht" if high_coh else "omp"


# --------------------------------------------------------------- 精修
def polish_support(
    A: np.ndarray, y: np.ndarray, support: np.ndarray, max_rounds: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """支撑集 1-opt 单调精修（交换最弱支撑点与最大残差相关非支撑点）。

    每次交换**只在降低训练残差时接受**，故 ``||y - A_S x_S||^2`` 单调不增
    是本函数的可验证不变量。

    **重要边界（实测）**：本步骤在 ``m < n`` 时会**过拟合**——训练残差不是
    有效的模型选择信号。作为消融项保留（``use_polish``），**默认关闭**；
    开启时仅做 1 轮且要求相对下降超过 5% 才接受。
    """
    n = A.shape[1]
    sup = np.unique(np.asarray(support, dtype=int))
    if sup.size == 0:
        return sup, np.zeros(n)
    coef = np.linalg.lstsq(A[:, sup], y, rcond=None)[0]
    x = np.zeros(n)
    x[sup] = coef
    best = float(np.sum((y - A @ x) ** 2))
    for _ in range(max_rounds):
        r = y - A @ x
        mask = np.ones(n, dtype=bool)
        mask[sup] = False
        if not mask.any() or sup.size < 2:
            break
        j = int(np.flatnonzero(mask)[int(np.argmax(np.abs(A[:, mask].T @ r)))])
        i = int(sup[int(np.argmin(np.abs(coef)))])
        trial = np.union1d(np.setdiff1d(sup, [i]), [j])
        if trial.size == 0 or trial.size >= A.shape[0] - 1:
            break
        c_t = np.linalg.lstsq(A[:, trial], y, rcond=None)[0]
        x_t = np.zeros(n)
        x_t[trial] = c_t
        val = float(np.sum((y - A @ x_t) ** 2))
        if val < best * 0.95:  # 需相对下降 >5%，抑制噪声驱动的伪改善
            best, sup, x, coef = val, trial, x_t, c_t
        else:
            break
    return sup, x


# --------------------------------------------------------------- 旗舰
class CS_Fuse:  # noqa: N801 - 与系统名 SparseForge 的 CS-Fuse 品牌一致，刻意保留
    """CS-Fuse：压缩感知自适应恢复求解器（旗舰）。

    Parameters
    ----------
    use_kest:
        是否做真值无关的 k 估计（消融 A1）。关闭时 ``k_hint`` 为必需。
    use_route:
        是否按 (相干度, SNR) 路由求解器（消融 A2）。关闭时固定用 ``omp``。
    use_debias:
        是否做支撑集最小二乘去偏（消融 A3）。
    use_polish:
        是否做 1-opt 精修（消融 A4，默认 False —— 实测 m<n 下会过拟合）。
    fixed_solver:
        ``use_route=False`` 时的固定求解器。
    kest:
        k 估计器名。``"elbow"``（默认，唯一通过 2-fold 交叉验证的判据）/
        ``"discrepancy"``（不一致性原理，仅作消融对照）。
    elbow_eps:
        ``kest="elbow"`` 时的拐点阈值。默认取 :data:`cs.kselect.ELBOW_EPS`
        （0.25，由双向交叉验证的中值确定，非测试集单点最优）。
    """

    name = "cs_fuse"

    def __init__(
        self,
        use_kest: bool = True,
        use_route: bool = True,
        use_debias: bool = True,
        use_polish: bool = False,
        fixed_solver: str = "omp",
        k_grid: Sequence[int] = K_GRID,
        tau: float = 1.0,
        kest: str = "elbow",
        elbow_eps: float = ELBOW_EPS,
    ) -> None:
        self.use_kest = bool(use_kest)
        self.use_route = bool(use_route)
        self.use_debias = bool(use_debias)
        self.use_polish = bool(use_polish)
        self.fixed_solver = fixed_solver
        self.k_grid = tuple(int(v) for v in k_grid)
        self.tau = float(tau)
        self.kest = str(kest)
        self.elbow_eps = float(elbow_eps)
        if self.tau <= 0:
            raise NumericsError("tau 必须为正", tau=tau)
        if self.kest not in ("elbow", "discrepancy"):
            raise NumericsError(
                "未知 k 估计器", kest=self.kest, allowed=["elbow", "discrepancy"]
            )

    def solve(
        self, A: np.ndarray, y: np.ndarray, k: int | None = None, cfg: SolveConfig | None = None
    ) -> RecoveryResult:
        cfg = cfg or SolveConfig()
        t0 = time.perf_counter()
        m, n = A.shape

        # --- 阶段 1：k 估计（真值无关） -----------------------------------
        sigma = None
        if self.use_kest or k is None:
            if self.kest == "elbow":
                k_hat = elbow_k(A, y, self.k_grid, self.elbow_eps)
                sigma = estimate_sigma(A, y)
            else:
                k_hat, sigma = noise_discrepancy_k(A, y, self.k_grid, self.tau)
        else:
            k_hat = int(k)
        k_hat = int(np.clip(k_hat, 1, m - 2))
        k_used = int(k) if (k is not None and not self.use_kest) else k_hat

        # --- 阶段 2：路由 -------------------------------------------------
        coh = coherence(A)
        snr_db = estimate_snr_db(A, y, sigma)
        solver_name = route_solver(coh, snr_db, m, n) if self.use_route else self.fixed_solver

        sub = SolveConfig(
            k=k_used,
            max_iter=cfg.max_iter,
            tol=cfg.tol,
            lam=cfg.lam,
            damping=cfg.damping,
            debias=False,
        )
        x = self._run_solver(solver_name, A, y, k_used, sub)

        # --- 阶段 3：去偏 -------------------------------------------------
        n_supp_pre = int(np.count_nonzero(np.abs(x) > 1e-10))
        if self.use_debias:
            sup = np.flatnonzero(np.abs(x) > 1e-10)
            if sup.size:
                x = debias_refit(A, y, sup)

        # --- 阶段 4：精修（默认关闭） -------------------------------------
        n_swaps = 0
        if self.use_polish:
            sup = np.flatnonzero(np.abs(x) > 1e-10)
            if sup.size:
                sup2, x2 = polish_support(A, y, sup, 3)
                n_swaps = int(sup2.size != sup.size)
                x = x2

        check_finite(x)
        sup = np.flatnonzero(np.abs(x) > 1e-10)
        return RecoveryResult(
            x_hat=x,
            support=sup,
            residual_norm=float(np.linalg.norm(y - A @ x)),
            nmse=float("nan"),
            support_f1=float("nan"),
            n_iter=int(cfg.max_iter),
            converged=True,
            solver=self.name,
            elapsed_sec=time.perf_counter() - t0,
            meta={
                "k_used": k_used,
                "k_true_given": k is not None,
                "kest": self.kest if (self.use_kest or k is None) else "given",
                "elbow_eps": float(self.elbow_eps) if self.kest == "elbow" else float("nan"),
                "solver": solver_name,
                "coherence": float(coh),
                "coherence_excess": float(coh / max(random_coherence_baseline(m, n), _EPS)),
                "snr_db_est": float(snr_db),
                "sigma_hat": float(sigma) if sigma is not None else float("nan"),
                "n_supp_pre_debias": n_supp_pre,
                "n_supp": int(sup.size),
                "n_swaps": n_swaps,
            },
        )

    @staticmethod
    def _run_solver(
        name: str, A: np.ndarray, y: np.ndarray, k: int, cfg: SolveConfig
    ) -> np.ndarray:
        """按名执行；失败时降级到 omp（保证旗舰永不抛错）。"""
        table = {
            "omp": omp,
            "cosamp": cosamp,
            "iht": iht,
            "fista": fista,
            "amp": amp,
            "rw_fista": reweighted_fista,
            "block_omp": block_omp,
        }
        fn = table.get(name, omp)
        try:
            return np.asarray(fn(A, y, k, cfg).x_hat, dtype=np.float64)
        except Exception:
            return np.asarray(omp(A, y, k, cfg).x_hat, dtype=np.float64)


def estimate_snr_db(A: np.ndarray, y: np.ndarray, sigma: float | None = None) -> float:
    """估计 SNR（dB）：``20 log10(||A x_hat|| / ||noise||)``，真值无关。

    用 OMP(k=m//8) 的解估计信号能量，噪声能量取 ``m * sigma_hat^2``。
    """
    m = A.shape[0]
    sh = estimate_sigma(A, y) if sigma is None else float(sigma)
    x = omp(A, y, max(1, m // 8), SolveConfig(k=max(1, m // 8), max_iter=200)).x_hat
    sig = float(np.linalg.norm(A @ x))
    noise = np.sqrt(max(m - int(np.count_nonzero(np.abs(x) > 1e-10)), 1)) * max(sh, _EPS)
    return float(20.0 * np.log10(max(sig, _EPS) / max(noise, _EPS)))
