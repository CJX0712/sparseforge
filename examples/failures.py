"""examples.failures — 复现项目已记录的真实失败/负结果案例。

对应 CLI 子命令 ``python cli.py failures``。每条案例都是**可复现**的负面
证据（非实现缺陷，而是压缩感知在该设定下的结构性边界），佐证
``docs/model_card.md`` 与 ``references/pitfalls.md`` 的结论：

1. ``rw_fista`` 在 ``m<n`` 欠定下支撑集严重膨胀（k 不敏感）
2. ``stability_k`` 选的 k 劣于 ``elbow_k``（bootstrap 破坏相干结构）
3. ``polish_support`` 在 ``m<n`` 下把正确支撑改坏（过拟合）
4. 不一致性原理 ``tau=0.40`` 选的 k 偏小于 elbow（测试集过拟合，不泛化）

作者：晨星 · CJX0712
"""

from __future__ import annotations

import numpy as np

from core.config import SolveConfig
from core.seed import set_all
from cs.fusion import polish_support
from cs.kselect import K_GRID, elbow_k, noise_discrepancy_k, stability_k
from cs.reweighted import reweighted_fista
from cs.solvers import omp
from data.generators import make_instance


def run_failure_cases() -> list[dict]:
    """返回复现的失败案例列表（每条含 kind/title/expected/observed）。"""
    set_all(20261005)
    cases: list[dict] = []

    # 1) rw_fista 支撑集膨胀（m<n 欠定，k 不敏感）
    inst = make_instance(
        "coherent", m=96, n=256, k=12, snr_db=18.0, coherence=0.9, seed=20261005
    )
    xr = reweighted_fista(inst.A, inst.y, inst.k_true, SolveConfig(k=inst.k_true, max_iter=300))
    sup = np.flatnonzero(np.abs(xr.x_hat) > 1e-10)
    cases.append(
        {
            "kind": "rw_fista_support_bloat",
            "title": "rw_fista 在 m<n 欠定下支撑集膨胀",
            "expected": f"n_supp ≈ k_true={inst.k_true}",
            "observed": f"n_supp={int(sup.size)} (膨胀 {sup.size / max(inst.k_true, 1):.1f}x)",
        }
    )

    # 2) stability_k 与 elbow_k 选的 k 不同（聚合层面 stability 劣于 elbow，单实例可相近）
    inst2 = make_instance("wellcond", m=48, n=128, k=6, snr_db=25.0, seed=9)
    ke = elbow_k(inst2.A, inst2.y, K_GRID, 0.25)
    ks, _ = stability_k(inst2.A, inst2.y, n_boot=8)
    re = omp(inst2.A, inst2.y, ke, SolveConfig(k=ke, max_iter=200))
    rs = omp(inst2.A, inst2.y, ks, SolveConfig(k=ks, max_iter=200))
    e_e = inst2.score(re.x_hat)[0]
    e_s = inst2.score(rs.x_hat)[0]
    cases.append(
        {
            "kind": "stability_k_differs_from_elbow",
            "title": "stability_k 与 elbow_k 选的 k 不同（单实例误差可相近，聚合层面 stability 劣于 elbow）",
            "expected": f"elbow_k={ke}, stability_k={ks}",
            "observed": f"elbow NMSE={e_e:.4e} | stability NMSE={e_s:.4e}",
        }
    )

    # 3) polish_support 在 m<n 下把正确支撑改坏（过拟合）
    gen = np.random.default_rng(31)
    m, n = 40, 100
    A = gen.standard_normal((m, n))
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    true_s = np.sort(gen.choice(n, 5, replace=False))
    y = A[:, true_s] @ gen.standard_normal(5) + 0.01 * gen.standard_normal(m)
    sup_out, _ = polish_support(A, y, true_s, max_rounds=3)
    changed = not np.array_equal(sup_out, true_s)
    cases.append(
        {
            "kind": "polish_overfit",
            "title": "polish_support 在 m<n 下把正确支撑改坏（过拟合）",
            "expected": "支撑集保持不变",
            "observed": "支撑集被改动（过拟合）" if changed else "未改动（罕见）",
        }
    )

    # 4) 不一致性原理 tau=0.40 偏小于 elbow（测试集过拟合，不泛化）
    inst3 = make_instance("wellcond", m=96, n=256, k=12, snr_db=18.0, seed=3)
    k40, _ = noise_discrepancy_k(inst3.A, inst3.y, tau=0.40)
    ke3 = elbow_k(inst3.A, inst3.y, K_GRID, 0.25)
    cases.append(
        {
            "kind": "discrepancy_tau_overfit",
            "title": "不一致性原理 tau=0.40 选的 k 偏小于 elbow（过保守/测试集过拟合）",
            "expected": f"elbow_k={ke3}",
            "observed": f"tau=0.40 -> k={k40}",
        }
    )

    return cases


if __name__ == "__main__":
    for c in run_failure_cases():
        print(f"  [{c['kind']}] {c['title']}: {c['expected']} -> {c['observed']}")
    print(f"\n{len(run_failure_cases())} failure cases reproduced.")
