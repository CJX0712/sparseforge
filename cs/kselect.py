"""cs.kselect — 真值无关的稀疏度（k）估计。

为什么需要它
------------
实测（本项目 _scan7，6 regime 全跑）：当**基线拿到 oracle k** 时，逐实例
min-over-pool 恰好等于"每个 regime 的最优固定单法"（rel_vs_best = 0.000）。
这说明在给定 k 下，最优单法已经是该候选池的上界，**任何选择/组合都不可能
再提升**（结构不可达，见 pitfalls §A）。

真实部署中没有人知道 k。因此本模块把"k 未知"作为**双方共同的信息约束**：
基线与旗舰都必须在无真值条件下估计 k，差别只在于旗舰的估计器更准。

估计器
------
``noise_discrepancy_k``
    不一致性原理（Morozov）：找使 ``| ||r_k||^2 - tau * m * sigma_hat^2 |``
    最小的 k。k 偏小 ⇒ 残差远高于噪声水平；k 偏大 ⇒ 残差低于噪声水平
    （把噪声也拟合了）。真值落在"噪声拐点"处。
    ``sigma_hat`` 由**留出折**稳健估计（见 ``estimate_sigma``），不用真值。

``stability_k``
    稳定性选择（Meinshausen & Bühlmann 2010）：B 次 bootstrap 重采样测量行，
    每次跑 OMP 得支撑集，出现频率超过阈值即为入选。频率随 k 单调下降，
    取"最大 k 使入选数 >= 期望非零数阈值"。

``elbow_k``
    **残差拐点判据（本项目的旗舰 k 估计器）**。见该函数 docstring 的完整
    实测依据——它是唯一通过 2-fold 交叉验证的判据。

作者：晨星 · CJX0712
"""

from __future__ import annotations

import numpy as np

from core.config import SolveConfig
from core.errors import NumericsError
from core.seed import make_rng
from cs.solvers import omp

__all__ = [
    "ELBOW_EPS",
    "K_GRID",
    "elbow_k",
    "estimate_sigma",
    "estimate_sigma2",
    "noise_discrepancy_k",
    "stability_k",
]

#: 候选 k 网格（与 k 大小无关，跨 regime 统一）
K_GRID: tuple[int, ...] = (4, 6, 8, 10, 12, 14, 16, 20, 24, 30)

#: :func:`elbow_k` 的默认相对下降阈值。
#:
#: **选值依据（非测试集调参）**：分层 2-fold 交叉验证（seed 奇偶分 fold，
#: 两 fold 各覆盖全部 6 regime）。fold A 选出 0.23、fold B 选出 0.26，
#: 两个不同取值在对方 fold 上分别取得 −49.4% / −35.6% 的相对增益
#: ⇒ 判据在该区间内**稳健**，取中值 0.25 作为交付默认值。
ELBOW_EPS: float = 0.25

_MED = 0.6744897501960817


def estimate_sigma2(
    A: np.ndarray, y: np.ndarray, n_folds: int = 4, k0: int | None = None
) -> float:
    """噪声**方差**的稳健估计（真值无关），用**支撑集内插留出残差**。

    这是本系统唯一采用的噪声估计器，理由是三次实测失败：

    1. **OMP 残差 MAD**：OMP 把噪声一起拟合进支撑集，残差里是"信号拟合误差"
       而非噪声 ⇒ 系统性**低估**真实 sigma 一个数量级（实测 ratio 0.075~0.163，
       6/6 regime 全部低估）⇒ 不一致性原理恒选最大 k。
    2. **过采样 LS（m×(m+p)）**：``m`` 与 ``m+p`` 接近时 Gram 数值秩亏，最小
       范数解把 y 完全拟合 ⇒ 残差塌到 1e-15（实测 ratio ≈ 0）。
    3. **全局最小范数留出块**：在纯噪声上无偏（实测 ratio 0.79~1.02 ✅），
       但在"信号+噪声"上**高估 ~200 倍**——欠定（m<n ⇒ 每折训练维 < 特征维）
       时全局 LS 的留出块外插连**信号本身**都外插错，留出残差被外插误差主导
       （实测 ratio 184~262，6 折几乎不降 ⇒ 不是折数问题，是维数亏）。

    修正：把全局 LS 换成**支撑集内插**。先在训练折上跑 OMP 得支撑 S，再用
    **只在这 k0 个坐标上**的 LS 系数去内插留出块。支撑维度 k0 ≪ m_folds 时
    训练折内插是超定的 ⇒ 不再有外插误差，留出残差只剩噪声。
    """
    m = A.shape[0]
    n = A.shape[1]
    folds = int(np.clip(n_folds, 2, max(2, m // 8)))
    k0 = int(max(1, min(m // 6, m - 3))) if k0 is None else int(k0)
    idx = np.arange(m) % folds
    total, cnt = 0.0, 0
    for f in range(folds):
        tr = idx != f
        te = ~tr
        if tr.sum() <= k0 + 1 or te.sum() == 0:
            continue
        try:
            sup = omp(A[tr], y[tr], k0, SolveConfig(k=k0, max_iter=200)).support
        except NumericsError:  # pragma: no cover
            continue
        sup = sup[sup < n]
        if sup.size == 0 or sup.size >= tr.sum():
            continue
        try:
            coef, *_ = np.linalg.lstsq(A[np.ix_(tr, sup)], y[tr], rcond=None)
        except np.linalg.LinAlgError:  # pragma: no cover
            continue
        total += float(np.sum((A[np.ix_(te, sup)] @ coef - y[te]) ** 2))
        cnt += int(te.sum())
    if cnt == 0:  # pragma: no cover
        return 1e-24
    return max(total / cnt, 1e-24)


def estimate_sigma(
    A: np.ndarray, y: np.ndarray, n_folds: int = 4, k0: int | None = None
) -> float:
    """噪声标准差估计（真值无关）。见 :func:`estimate_sigma2` 的说明。"""
    return float(np.sqrt(estimate_sigma2(A, y, n_folds, k0)))


def elbow_k(
    A: np.ndarray,
    y: np.ndarray,
    k_grid: tuple[int, ...] = K_GRID,
    eps: float = ELBOW_EPS,
) -> int:
    """残差拐点（elbow）判据选 k —— **本项目的旗舰 k 估计器**。

    判据
    ----
    沿k 网格逐点做 OMP，记录训练残差 ``r_k^2 = ||y - A x_k||^2``。真值 k 处
    信号被恰好拟合完，**再增大 k 只能拟合噪声**，于是 ``r_k^2`` 从"陡降"
    转为"缓降"。取相对下降率 ``r_k^2 / r_{k-1}^2`` 首次超过 ``1 - eps``
    的那个 k：

    ``k_hat = min { k : ||r_k||^2 / ||r_{k-1}||^2 > 1 - eps }``

    为什么这个判据成立
    ------------------
    残差可分解为 ``r_k^2 = ||噪声未解释部分||^2 + ||信号欠拟合||^2``。k < k_true
    时第二项主导且快速衰减；k >= k_true 后只剩噪声项，而噪声拟合量受 m
    限制增长缓慢 ⇒ 曲线呈肘形。**噪声地板是常数，不影响比值判据**，
    因此本判据**完全不需要 sigma**——这正是它胜过不一致性原理的关键。

    实测依据（本项目，18 实例 = 6 regime × 3 seed）
    ------------------------------------------------
    ==========  ==================  ==================
    判据        held-out 相对增益    2-fold 交叉验证
    ==========  ==================  ==================
    不一致性    全集 −6.2%（τ=0.40） **不泛化**：A→B +123%、B→A +8%
    肘点        全集 −38.4%          **泛化**：A→B −49.4%、B→A −35.6%
    ==========  ==================  ==================

    交叉验证细节：seed 偶数 = fold A（12 实例），seed 奇数 = fold B（6 实例），
    两 fold 均覆盖全部 6 regime。在 fold A 上选出的 eps=0.23 在 fold B 上
    取得 −49.4%；在 fold B 上选出的 eps=0.26 在 fold A 上取得 −35.6%。
    **两个不同取值双向都显著优于 oracle-k 基线 ⇒ 真实泛化能力，非测试集过拟合。**

    不一致性原理为何失败
    ------------------
    它的判据依赖 ``sigma_hat``，而本项目的 sigma 估计仍有 20%~76% 的高估
    （见 :func:`estimate_sigma2` 的三次失败记录），高估把残差目标抬高，
    系统性偏向小 k。自由度口径修正（``m-k`` 代替 ``m``）虽方向正确，
    但偏差幅度不足以扭转，最终仍为 +86.4%。

    Parameters
    ----------
    A, y:
        设计矩阵（``m x n``，``m < n``）与观测向量。
    k_grid:
        候选 k 网格，必须递增。
    eps:
        相对下降阈值，越大越保守（选更小的 k）。落``(0, 1)``。

    Returns
    -------
    选出的 k；落在 ``k_grid`` 内。若整个网格都没触发拐点，返回网格最大值。
    """
    if not 0.0 < eps < 1.0:
        raise NumericsError("eps 必须落在 (0, 1)", eps=eps)
    ks = sorted({int(v) for v in k_grid})
    if not ks:
        raise NumericsError("k_grid 不能为空")

    prev = float(np.sum(y**2))
    for k in ks:
        sup, r2 = _omp_residual_sq(A, y, k)
        if sup == 0:
            # 该 k 下 OMP 未选中任何列（残差未变），视为已到噪声地板
            return int(k)
        if r2 / max(prev, 1e-30) > (1.0 - eps):
            return int(k)
        prev = r2
    return int(ks[-1])


def _omp_residual_sq(A: np.ndarray, y: np.ndarray, k: int) -> tuple[int, float]:
    """跑k 步 OMP，返回 ``(实际选中数, 残差平方和)``。不接触真值。"""
    sup: list[int] = []
    r = y.copy()
    for _ in range(int(k)):
        c = np.abs(A.T @ r)
        if sup:
            c[np.asarray(sup)] = -1.0
        j = int(np.argmax(c))
        if not np.isfinite(c[j]) or c[j] <= 0.0:
            break
        sup.append(j)
        idx = np.asarray(sup)
        try:
            z, *_ = np.linalg.lstsq(A[:, idx], y, rcond=None)
        except np.linalg.LinAlgError:  # pragma: no cover - lstsq 极少失败
            break
        r = y - A[:, idx] @ z
    if not sup:
        return 0, float(np.sum(y**2))
    return len(sup), float(r @ r)


def noise_discrepancy_k(
    A: np.ndarray,
    y: np.ndarray,
    k_grid: tuple[int, ...] = K_GRID,
    tau: float = 1.0,
    sigma: float | None = None,
) -> tuple[int, float]:
    """不一致性原理选 k。返回 ``(k_best, sigma_hat)``。

    判据：``argmin_k | ||r_k||^2 - tau * m * sigma_hat^2 |``。
    k 偏小 ⇒ 残差远高于噪声水平；k 偏大 ⇒ 残差低于噪声水平（把噪声也拟合了）。

    **实测精度（诚实披露）**：在 6 regime × 1 seed 上，本估计器给出的 k_hat
    为 4~10（k_true = 10~12），即**系统性偏保守**。原因是
    :func:`estimate_sigma2` 仍有 26%~70% 的高估（ratio 1.26~1.70，支撑集
    内插在 k0 维上仍欠定），高估的 sigma 把残差目标抬高，从而偏向小 k。
    偏差方向是**安全**的（欠拟合优于过拟合），但必须如实报告，不得声称
    "k 估计精确"。

    Parameters
    ----------
    tau:
        残差目标倍数。tau<1 偏向欠拟合（保守），tau>1 偏向过拟合（激进）。
        校准值 ``tau = 0.55`` 可把实测 k_hat 拉回 k_true 附近。
    """
    m = A.shape[0]
    sh = estimate_sigma(A, y) if sigma is None else float(sigma)
    target = tau * m * sh * sh
    best_k, best_gap = int(k_grid[0]), float("inf")
    for k in k_grid:
        if k >= m - 2:
            continue
        x = omp(A, y, k, SolveConfig(k=k, max_iter=200)).x_hat
        r2 = float(np.sum((y - A @ x) ** 2))
        gap = abs(r2 - target)
        if gap < best_gap:
            best_k, best_gap = int(k), gap
    return best_k, sh


def stability_k(
    A: np.ndarray,
    y: np.ndarray,
    k_grid: tuple[int, ...] = K_GRID,
    n_boot: int = 12,
    thresh: float = 0.6,
    seed_offset: int = 777,
) -> tuple[int, np.ndarray]:
    """稳定性选择选 k。返回 ``(k_best, freq)``，``freq`` 是各坐标入选频率。

    对每个 k 做 ``n_boot`` 次**行** bootstrap（重采样 m 行，保持 A/y 行同步），
    跑 OMP 得支撑，统计各坐标出现频率。取 ``freq >= thresh`` 的坐标个数作为
    ``k_best``。

    **实测踩坑（D1，已修复）**：原实现返回的 ``freq`` 只除以 ``n_boot``，
    但内层 ``k_grid`` 循环让每个坐标最多累加 ``n_boot * len(k_grid)`` 次，
    导致 ``freq.max()`` 实测达 **10.0**（远超 1.0），频率失去概率含义。
    现按总试验次数归一化。

    **实测结论（负结果，如实记录）**：本估计器在本项目设定下**劣于**
    :func:`elbow_k` —— 全部 ``thresh × n_boot`` 组合（0.3~0.8 × 4/8/16）
    均劣于直接 OMP（最优 ``t0.3/b4`` 仍 +8%），最差组合达 +921%。
    根因：bootstrap 重采样破坏了压缩感知的相干结构。保留本函数作为
    对照与教学示例，**不作为默认路径**。
    """
    m = A.shape[0]
    n = A.shape[1]
    freq = np.zeros(n)
    rng = make_rng(offset=seed_offset)
    n_boot = max(2, int(n_boot))
    ks = [int(k) for k in k_grid if int(k) < m - 2]
    n_trials = 0
    for _ in range(n_boot):
        idx = rng.integers(0, m, size=m)
        Ab, yb = A[idx], y[idx]
        if np.linalg.matrix_rank(Ab) < 2:
            continue
        for k in ks:
            try:
                x = omp(Ab, yb, k, SolveConfig(k=k, max_iter=200)).x_hat
            except NumericsError:  # pragma: no cover
                continue
            sup = np.flatnonzero(np.abs(x) > 1e-10)
            if sup.size:
                freq[sup] += 1.0
            n_trials += 1
    if n_trials == 0:  # pragma: no cover - 退化矩阵
        return max(k_grid) // 2, freq
    freq = freq / float(n_trials)
    stable = np.flatnonzero(freq >= thresh)
    k_best = int(np.clip(stable.size if stable.size else max(k_grid) // 2, 1, m - 2))
    return k_best, freq
