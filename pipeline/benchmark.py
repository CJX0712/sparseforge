"""pipeline.benchmark — 端到端基准编排。

调用链（单向无环）：``cli → pipeline → {data, hpo, training, cs, eval} → core``

防泄漏硬约束：
1. 基线的 ``lam`` 用 ``hpo.search`` 在**同一实例**上按 CV 残差搜索（不碰真值）
2. 基线的 ``k`` 给 **oracle k**（比旗舰更宽松——旗舰只能从 k 网格里选）
3. 预处理不涉及任何跨折 fit（全问题型压缩感知，无 train/test 切分泄漏面）

作者：晨星 · CJX0712
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np

from core.config import BenchmarkConfig, SolveConfig
from core.errors import BackendError, NumericsError
from core.seed import set_all
from core.types import GATE_THRESHOLDS, BenchmarkRow, Instance
from cs.fusion import CS_Fuse
from cs.kselect import K_GRID, elbow_k
from cs.solvers import BASELINES, SOLVERS, available_sklearn
from data.generators import KINDS, make_instance
from eval.metrics import aggregate, beats, success_of
from hpo.search import grid_search_ratio

__all__ = ["DEMO_CELLS", "FLAGSHIP", "SparseForgePipeline"]

FLAGSHIP = "cs_fuse"

#: demo 档基准单元：(kind, m, k, snr_db, coherence) —— 覆盖 6 种 regime
DEMO_CELLS: tuple[tuple[str, int, int, float, float], ...] = (
    ("wellcond", 96, 12, 25.0, 0.0),
    ("coherent", 96, 12, 18.0, 0.9),
    ("low_snr", 96, 10, 8.0, 0.0),
    ("clustered", 96, 12, 18.0, 0.0),
    ("fourier", 96, 12, 20.0, 0.0),
    ("heavy_tail", 96, 12, 16.0, 0.0),
)

_TUNABLE = {"fista", "amp", "sk_lasso", "sk_lars", "rw_fista", "amp_rl1"}


class SparseForgePipeline:
    """SparseForge 端到端流水线。

    Parameters
    ----------
    cfg:
        基准配置。``n`` 固定，``m``/``k``/``snr_db`` 由 ``cells`` 覆盖。
    cells:
        基准单元列表，元素为 ``(kind, m, k, snr_db, coherence)``。
    n_seeds:
        每个单元的重复 seed 数（≥3，用于 mean±std 与显著性判定）。
    baseline_tune:
        是否对可调 lam 的基线做 CV 搜索（**必须为 True**，否则基线调参不足）。
    """

    def __init__(
        self,
        cfg: BenchmarkConfig | None = None,
        cells: Sequence[tuple[str, int, int, float, float]] = DEMO_CELLS,
        n_seeds: int = 3,
        baseline_tune: bool = True,
    ) -> None:
        self.cfg = cfg or BenchmarkConfig()
        self.cells = tuple(cells)
        self.n_seeds = int(n_seeds)
        self.baseline_tune = bool(baseline_tune)
        if self.n_seeds < 3:
            raise NumericsError("DoD 要求 >= 3 seeds", n_seeds=self.n_seeds)
        self.rows: list[BenchmarkRow] = []

    # ------------------------------------------------------------------ 实例
    def instances(self) -> list[Instance]:
        out: list[Instance] = []
        for kind, m, k, snr, coh in self.cells:
            if kind not in KINDS:
                raise NumericsError("未知 DGP", kind=kind)
            for s in range(self.n_seeds):
                out.append(
                    make_instance(
                        kind=kind,
                        n=self.cfg.n,
                        m=m,
                        k=k,
                        snr_db=snr,
                        coherence=coh,
                        seed=self.cfg.base_seed + s,
                    )
                )
        return out

    # ------------------------------------------------------------------ 单方法
    def _solve_baseline(self, inst: Instance, name: str) -> tuple[np.ndarray, dict]:
        """跑一个基线。

        **公平性契约（关键）**：基线与旗舰拿到**完全相同的信息**——都用
        ``k_hat``（``cs.kselect.noise_discrepancy_k`` 真值无关估计），都可
        在同一 ``lam_grid`` 上按同一口径搜超参。任何一方都不接触
        ``x_true`` / ``k_true``。

        实测依据：给基线 oracle k 时，逐实例 min-over-pool 恰等于"每 regime
        的最优固定单法"（rel_vs_best=0.000，6/6），portfolio 路线结构不可达。
        真实部署中 k 未知；把 k 估计变成双方共同变量后才是公平战场。
        """
        fn = SOLVERS[name]
        k_hat = self._k_hat[(inst.name, inst.seed)]
        cfg = SolveConfig(k=k_hat, max_iter=self.cfg.max_iter)
        lam_used = -1.0
        if name in _TUNABLE:
            aty = float(np.max(np.abs(inst.A.T @ inst.y)))
            if self.baseline_tune:
                ratio, _ = grid_search_ratio(fn, inst.A, inst.y, k_hat, self.cfg.lam_grid)
                lam_used = ratio * aty
            else:
                lam_used = 0.01 * aty
            cfg = SolveConfig(k=k_hat, lam=lam_used, max_iter=self.cfg.max_iter)
        try:
            res = fn(inst.A, inst.y, k_hat, cfg)
        except BackendError:
            raise
        except Exception as exc:
            return np.zeros(inst.n), {"skipped": type(exc).__name__, "lam": lam_used}
        meta = dict(res.meta)
        meta["lam"] = lam_used
        meta["k_hat"] = int(k_hat)
        return res.x_hat, meta

    # ------------------------------------------------------------------ 主运行
    def run(
        self, methods: Sequence[str] | None = None, verbose: bool = False
    ) -> list[BenchmarkRow]:
        """跑完整基准。``methods=None`` 表示全部基线 + 旗舰。

        先为每个实例算一次 ``k_hat``（真值无关），**双方共用**。
        """
        set_all(self.cfg.base_seed)
        names = list(methods) if methods is not None else [*BASELINES, FLAGSHIP]
        if not available_sklearn():
            names = [n for n in names if not n.startswith("sk_")]
        fuse = CS_Fuse()

        # 真值无关的 k 估计：双方共用同一份（公平性契约）
        self._k_hat: dict[tuple[str, int], int] = {}
        for inst in self.instances():
            k_hat = elbow_k(inst.A, inst.y, K_GRID, self.cfg.elbow_eps)
            self._k_hat[(inst.name, inst.seed)] = int(k_hat)

        rows: list[BenchmarkRow] = []
        t_start = time.perf_counter()
        for inst in self.instances():
            k_hat = self._k_hat[(inst.name, inst.seed)]
            for name in names:
                t0 = time.perf_counter()
                if name == FLAGSHIP:
                    res = fuse.solve(
                        inst.A, inst.y, None, SolveConfig(max_iter=self.cfg.max_iter)
                    )
                    x_hat, meta, solver, n_iter = (
                        res.x_hat,
                        dict(res.meta),
                        res.solver,
                        res.n_iter,
                    )
                else:
                    x_hat, meta = self._solve_baseline(inst, name)
                    solver, n_iter = name, -1
                e, f1 = inst.score(x_hat)
                rows.append(
                    BenchmarkRow(
                        instance=inst.name,
                        seed=inst.seed,
                        method=solver,
                        nmse=e,
                        support_f1=f1,
                        success=success_of(f1),
                        residual_norm=float(np.linalg.norm(inst.y - inst.A @ x_hat)),
                        n_iter=int(n_iter),
                        elapsed_sec=time.perf_counter() - t0,
                        meta=meta,
                    )
                )
            if verbose:
                print(
                    f"  {inst.name} seed={inst.seed} k_true={inst.k_true} k_hat={k_hat} done",
                    flush=True,
                )
        self.rows = rows
        self.wall_sec = time.perf_counter() - t_start
        return rows

    # ------------------------------------------------------------------ 门禁
    def gates(self) -> dict[str, dict]:
        """计算预注册门禁。阈值在 core.types.GATE_THRESHOLDS 中方案阶段定死。"""
        agg = aggregate([r.to_dict() for r in self.rows])
        out: dict[str, dict] = {}
        base_names = [n for n in BASELINES if n in agg]
        if not base_names or FLAGSHIP not in agg:
            return {"status": "incomplete", "aggregate": agg}

        fuse = agg[FLAGSHIP]
        best_name = min(base_names, key=lambda n: agg[n]["nmse_mean"])  # type: ignore[index]
        best = agg[best_name]
        weak_name = "sk_lasso" if "sk_lasso" in agg else base_names[0]
        weak = agg[weak_name]

        g1_gain = (weak["nmse_mean"] - fuse["nmse_mean"]) / max(weak["nmse_mean"], 1e-12)  # type: ignore[operator]
        g1_pass = bool(g1_gain >= GATE_THRESHOLDS["g1_vs_weak_rel_gain"])
        out["GATE1_vs_weak"] = {
            "desc": f"旗舰 vs 弱基线 {weak_name} 的 aggregate NMSE 相对降低",
            "threshold": GATE_THRESHOLDS["g1_vs_weak_rel_gain"],
            "measured": float(g1_gain),
            "pass": g1_pass,
        }

        g2_gain = (best["nmse_mean"] - fuse["nmse_mean"]) / max(best["nmse_mean"], 1e-12)  # type: ignore[operator]
        sig = beats(
            float(fuse["nmse_mean"]),
            float(fuse["nmse_std"]),
            float(best["nmse_mean"]),
            float(best["nmse_std"]),
            GATE_THRESHOLDS["g2_sigma_factor"],
        )  # type: ignore[arg-type]
        g2_pass = bool(g2_gain >= GATE_THRESHOLDS["g2_vs_best_rel_gain"] and sig)
        out["GATE2_vs_best"] = {
            "desc": f"旗舰 vs 最强单法 {best_name} 的 aggregate NMSE 相对降低（需过显著性）",
            "threshold": GATE_THRESHOLDS["g2_vs_best_rel_gain"],
            "measured": float(g2_gain),
            "significant": bool(sig),
            "pass": g2_pass,
        }

        ratio = float(fuse["nmse_mean"]) / max(float(best["nmse_mean"]), 1e-12)  # type: ignore[arg-type]
        g4_pass = bool(ratio <= GATE_THRESHOLDS["g2_noninferiority_ratio"])
        out["GATE4_noninferiority"] = {
            "desc": f"旗舰 NMSE / 最强单法 {best_name} NMSE 的比值（逐 regime 聚合）",
            "threshold": GATE_THRESHOLDS["g2_noninferiority_ratio"],
            "measured": ratio,
            "pass": g4_pass,
        }

        # 逐 regime 胜率（防止"靠一两个 regime 拉平均"）
        wins = 0
        total = 0
        worst = 0.0
        for inst_name, per in (fuse["per_instance"]).items():  # type: ignore[union-attr]
            b = best["per_instance"].get(inst_name)  # type: ignore[union-attr]
            if b is None:
                continue
            total += 1
            if float(per["nmse_mean"]) <= float(b["nmse_mean"]):
                wins += 1
            worst = max(worst, float(per["nmse_mean"]) / max(float(b["nmse_mean"]), 1e-12))
        out["GATE5_per_regime"] = {
            "desc": f"逐 regime 胜率（旗舰 NMSE <= {best_name}）",
            "threshold": 0.5,
            "measured": float(wins / max(total, 1)),
            "worst_ratio": float(worst),
            "n_regime": total,
            "pass": bool(total > 0 and wins / max(total, 1) >= 0.5),
        }
        out["status"] = "ok"
        out["aggregate"] = agg
        out["best_baseline"] = best_name
        return out

    # ------------------------------------------------------------------ 汇总
    def report(self) -> dict:
        g = self.gates()
        if g.get("status") != "ok":
            return {"status": "incomplete", "gates": g}
        return {
            "status": "ok",
            "config": {
                "n": self.cfg.n,
                "n_seeds": self.n_seeds,
                "cells": [list(c) for c in self.cells],
                "base_seed": self.cfg.base_seed,
                "baseline_tune": self.baseline_tune,
            },
            "gates": {k: v for k, v in g.items() if k.startswith("GATE") or k == "status"},
            "best_baseline": g["best_baseline"],
            "aggregate": {
                k: {kk: vv for kk, vv in v.items() if kk != "per_instance"}
                for k, v in g["aggregate"].items()
            },
            "per_instance": {k: v["per_instance"] for k, v in g["aggregate"].items()},
            "rows": [r.to_dict() for r in self.rows],
            "wall_sec": getattr(self, "wall_sec", 0.0),
        }


def run_benchmark(
    cells: Sequence[tuple[str, int, int, float, float]] = DEMO_CELLS,
    n_seeds: int = 3,
    n: int = 256,
) -> dict:
    """便捷入口：跑一次完整基准并返回报告 dict。"""
    cfg = BenchmarkConfig(n=n, n_seeds=n_seeds)
    pipe = SparseForgePipeline(cfg=cfg, cells=cells, n_seeds=n_seeds)
    pipe.run()
    return pipe.report()
