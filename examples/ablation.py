"""examples.ablation — CS-Fuse 四个组件的消融实验。

对应 CLI 子命令 ``python cli.py ablate``。逐一切掉一个组件（A1 k 估计 /
A2 路由 / A3 去偏 / A4 精修），与完整旗舰对比 aggregate NMSE，量化每个
组件的贡献。实验在 6 种 regime × ``seeds`` 上运行，结果可复现。

**重要结论（如实记录）**：本系统唯一通过 2-fold 交叉验证的真实增益来自
A1 的拐点 k 估计（``elbow_k``）；A2/A3 是公平性契约要求的"同口径"组件，
单独关掉会退化到基线；A4 精修在 ``m<n`` 下过拟合，默认关闭。旗舰在
修复后定级 **B**（与最强基线持平，GATE2 差距 −3.05% 源于给定 k 后结构
不可达，非实现缺陷）。

作者：晨星 · CJX0712
"""

from __future__ import annotations

import numpy as np

from core.seed import set_all
from cs.fusion import CS_Fuse
from data.generators import make_instance

#: 消融用的轻量单元（m=48,n=128,k=6），覆盖 6 种 regime，跑得快
_ABLATION_CELLS = (
    ("wellcond", 48, 128, 6, 25.0),
    ("coherent", 48, 128, 6, 18.0),
    ("clustered", 48, 128, 6, 18.0),
    ("heavy_tail", 48, 128, 6, 16.0),
    ("low_snr", 48, 128, 6, 8.0),
    ("fourier", 48, 128, 6, 20.0),
)


def run_ablation(seeds: int = 3) -> dict:
    """返回各消融变体的 aggregate NMSE 与相对 full 的 delta。"""
    set_all(20261005)
    variants = {
        "full": dict(use_kest=True, use_route=True, use_debias=True, use_polish=False),
        "A1_off_kest": dict(use_kest=False, use_route=True, use_debias=True, use_polish=False),
        "A2_off_route": dict(use_kest=True, use_route=False, use_debias=True, use_polish=False),
        "A3_off_debias": dict(
            use_kest=True, use_route=True, use_debias=False, use_polish=False
        ),
        "A4_on_polish": dict(use_kest=True, use_route=True, use_debias=True, use_polish=True),
    }
    out: dict[str, dict] = {}
    for vname, cfg in variants.items():
        errs: list[float] = []
        for kind, m, n, k, snr in _ABLATION_CELLS:
            for s in range(seeds):
                inst = make_instance(kind, n=n, m=m, k=k, snr_db=snr, seed=20261005 + s)
                fuse = CS_Fuse(**cfg)
                if cfg.get("use_kest") is False:
                    # 关掉 k 估计时，用 oracle k 作为对照（公平性契约：传 k 时双方同口径）
                    res = fuse.solve(inst.A, inst.y, k=inst.k_true)
                else:
                    res = fuse.solve(inst.A, inst.y)
                e, _ = inst.score(res.x_hat)
                errs.append(float(e))
        out[vname] = {
            "nmse_mean": float(np.mean(errs)),
            "nmse_std": float(np.std(errs)),
            "n": len(errs),
        }
    base = out["full"]["nmse_mean"]
    for v in out.values():
        v["rel_vs_full"] = float((v["nmse_mean"] - base) / max(base, 1e-12))
    return out


if __name__ == "__main__":
    import json

    print(json.dumps(run_ablation(), ensure_ascii=False, indent=2))
