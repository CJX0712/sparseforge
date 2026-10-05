"""examples.run_demo — 端到端演示，产出 ``benchmark.json``。

对应 CLI 子命令 ``python cli.py demo``。跑完整基准并把聚合报告写入 JSON，
作为门禁判定的可复现证据。

作者：晨星 · CJX0712
"""

from __future__ import annotations

import json
from pathlib import Path

from core.seed import set_all
from pipeline.benchmark import DEMO_CELLS, SparseForgePipeline


def run_demo(out_path: str = "benchmark.json", seeds: int = 3) -> int:
    """跑完整基准并将报告写入 ``out_path``。

    Returns
    -------
    退出码（0 表示成功）。
    """
    set_all(20261005)
    pipe = SparseForgePipeline(n_seeds=seeds, cells=DEMO_CELLS)
    pipe.run(verbose=False)
    rep = pipe.report()
    Path(out_path).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    # 控制台摘要
    agg = rep.get("aggregate", {})
    fuse = agg.get("cs_fuse")
    if fuse is not None:
        print(
            f"[demo] cs_fuse NMSE_mean={float(fuse['nmse_mean']):.4e} "
            f"F1_mean={float(fuse['f1_mean']):.4f}"
        )
    gates = rep.get("gates", {})
    n_pass = sum(1 for v in gates.values() if isinstance(v, dict) and v.get("pass"))
    print(f"[demo] gates PASS={n_pass}/{len([k for k in gates if k.startswith('GATE')])}")
    print(f"[demo] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_demo())
