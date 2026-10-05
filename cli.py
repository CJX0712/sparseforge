"""cli — SparseForge 命令行入口。

子命令：
    demo        端到端演示（生成 benchmark.json）
    bench       只跑基准并打印表格
    ablate      消融实验
    failures    失败案例
    check       确定性自检（同 seed 两次运行核心指标逐位一致）

作者：晨星
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if sys.platform == "win32":  # pragma: no cover
    import contextlib

    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8")

from core.seed import set_all
from pipeline.benchmark import DEMO_CELLS, SparseForgePipeline

__all__ = ["build_parser", "main"]


def _print_table(title: str, agg: dict, top: int = 20) -> None:
    print(f"\n=== {title} ===")
    print(f"{'method':<12}{'NMSE mean':>14}{'std':>12}{'F1 mean':>10}{'succ':>8}{'n':>5}")
    items = sorted(agg.items(), key=lambda kv: float(kv[1]["nmse_mean"]))
    for name, v in items[:top]:
        print(
            f"{name:<12}{float(v['nmse_mean']):>14.6e}{float(v['nmse_std']):>12.3e}"
            f"{float(v['f1_mean']):>10.4f}{float(v['success_rate']):>8.3f}{int(v['n']):>5}"
        )


def _cmd_bench(args: argparse.Namespace) -> int:
    pipe = SparseForgePipeline(n_seeds=args.seeds, cells=DEMO_CELLS)
    pipe.run(verbose=args.verbose)
    rep = pipe.report()
    _print_table("SparseForge benchmark (aggregate over regimes x seeds)", rep["aggregate"])
    print("\n=== Gates ===")
    for k, v in rep["gates"].items():
        if not k.startswith("GATE"):
            continue
        mark = "PASS" if v["pass"] else "FAIL"
        extra = f" significant={v['significant']}" if "significant" in v else ""
        print(f"  [{mark}] {k}: measured={v['measured']:.4f} threshold={v['threshold']}{extra}")
        print(f"         {v['desc']}")
    print(f"\nwall_sec = {rep['wall_sec']:.2f}")
    if args.out:
        Path(args.out).write_text(
            json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"wrote {args.out}")
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    """确定性自检：同 seed 跑两次，比较核心指标是否逐位一致。"""
    keys = ("instance", "seed", "method", "nmse", "support_f1", "success", "residual_norm")
    runs = []
    for _ in range(2):
        p = SparseForgePipeline(n_seeds=args.seeds)
        rows = p.run()
        runs.append([tuple(r.to_dict()[k] for k in keys) for r in rows])
    same = runs[0] == runs[1]
    print(f"determinism rows={len(runs[0])} bit_identical={same}")
    if not same:
        for a, b in zip(runs[0], runs[1], strict=False):
            if a != b:
                print("  first diff:", a, "vs", b)
                break
    return 0 if same else 1


def _cmd_ablate(args: argparse.Namespace) -> int:
    from examples.ablation import run_ablation

    out = run_ablation(seeds=args.seeds)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def _cmd_failures(args: argparse.Namespace) -> int:
    from examples.failures import run_failure_cases

    cases = run_failure_cases()
    for c in cases:
        print(f"  [{c['kind']}] {c['title']}: {c['expected']} -> {c['observed']}")
    print(f"\n{len(cases)} failure cases reproduced.")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    from examples.run_demo import run_demo

    return run_demo(out_path=args.out, seeds=args.seeds)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sparseforge", description="SparseForge CLI · 作者 晨星")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("bench", help="跑基准并打印表")
    b.add_argument("--seeds", type=int, default=3)
    b.add_argument("--out", type=str, default="")
    b.add_argument("--verbose", action="store_true")
    b.set_defaults(func=_cmd_bench)

    c = sub.add_parser("check", help="确定性自检")
    c.add_argument("--seeds", type=int, default=3)
    c.set_defaults(func=_cmd_check)

    a = sub.add_parser("ablate", help="消融实验")
    a.add_argument("--seeds", type=int, default=3)
    a.set_defaults(func=_cmd_ablate)

    f = sub.add_parser("failures", help="失败案例")
    f.set_defaults(func=_cmd_failures)

    d = sub.add_parser("demo", help="端到端演示")
    d.add_argument("--out", type=str, default="benchmark.json")
    d.add_argument("--seeds", type=int, default=3)
    d.set_defaults(func=_cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    set_all()
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
