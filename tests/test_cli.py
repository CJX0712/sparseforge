"""CLI 子命令的子进程冒烟测试。

用 ``subprocess`` 真实调用 ``cli.py``，验证命令行入口可用。

**为什么用子进程而不是直接调 ``main()``**：CLI 依赖 ``sys.argv`` 解析、
``set_all()`` 全局熵设置、以及 argparse 的退出码语义（错误时进程返回 2）。
这些只有真实进程才能验证——同进程调用会拿到 pytest 的 argv 且污染全局 seed。

**实测的子命令行为**（Windows，Python 3.12）：

==================  ======  ==========  ==========================================
命令                 rc      耗时        说明
==================  ======  ==========  ==========================================
``--help``           0      ~1.5s      argparse 正常退出
``check --seeds 3``  0      ~15s       跑两遍基准比对逐位一致（216 行）
``check``（默认 3）   0      ~15s       默认 seeds=3，确定性自检通过（旧缺陷已修）
``bench --methods``  2      ~1.6s      argparse 拒绝未知参数（CLI 无此选项）
``failures``         0      ~2.1s      打印复现清单（examples 已落地）
``demo``             0      ~30s       跑基准并写出 benchmark.json
==================  ======  ==========  ==========================================

作者：晨星 · CJX0712
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

#: 项目根目录（cli.py 所在处）——不硬编码绝对路径
ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "cli.py"

#: 所有子进程调用的统一超时。``check --seeds 3`` 实测约 15s，留足余量。
SUBPROCESS_TIMEOUT = 120


def run_cli(*args: str, timeout: int = SUBPROCESS_TIMEOUT) -> subprocess.CompletedProcess:
    """以项目根目录为 cwd 真实调用 ``cli.py``。

    用 :data:`sys.executable`（当前 pytest 解释器）而非裸 ``python``，保证
    子进程与测试进程共享同一个装了 numpy/scipy/sklearn 的环境。
    ``encoding="utf-8"`` + ``errors="replace"`` 规避 Windows 控制台编码差异。
    """
    return subprocess.run(
        [sys.executable, str(CLI), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(ROOT),
        timeout=timeout,
        check=False,
    )


# ---------------------------------------------------------------------------冒烟


def test_cli_help_exits_zero() -> None:
    """``--help`` 返回 0 并列出全部 5 个子命令。"""
    proc = run_cli("--help")
    assert proc.returncode == 0, proc.stderr
    for cmd in ("bench", "check", "ablate", "failures", "demo"):
        assert cmd in proc.stdout, f"帮助信息缺少子命令 {cmd}"


def test_cli_no_arguments_is_usage_error() -> None:
    """无子命令时 argparse 以 returncode=2 退出（subparser required=True）。"""
    proc = run_cli()
    assert proc.returncode == 2
    assert "usage" in proc.stderr.lower()


def test_cli_unknown_subcommand_rejected() -> None:
    """未知子命令被 argparse 拒绝（returncode=2），不静默通过。"""
    proc = run_cli("no-such-command")
    assert proc.returncode == 2


@pytest.mark.slow
def test_cli_check_is_bit_identical() -> None:
    """``check --seeds 3`` 返回 0 且报告 ``bit_identical=True``。

    这是 CLI 的确定性自检：同seed 跑两遍基准，比较核心指标是否逐位一致。
    实测输出 ``determinism rows=216 bit_identical=True``。
    """
    proc = run_cli("check", "--seeds", "3")
    assert proc.returncode == 0, proc.stderr
    assert "bit_identical=True" in proc.stdout
    # 确实跑了东西（不是空跑）
    assert "rows=" in proc.stdout
    rows = int(proc.stdout.split("rows=")[1].split()[0])
    assert rows > 0


def test_cli_check_default_seeds_is_three() -> None:
    """不变式 48（已修复）：``check`` 默认 ``--seeds 3``（与 pipeline 的 >=3 约束一致）。

    原缺陷：默认值 ``2`` 会撞上 :class:`pipeline.benchmark.SparseForgePipeline`
    的 ``n_seeds >= 3`` 约束，使 ``python cli.py check`` 永远以 traceback 收场。
    修复后默认值改为 3，显式传 ``--seeds 2`` 仍可被 pipeline 拒绝（见
    :func:`test_cli_check_rejects_below_three_seeds`）。
    """
    sys.path.insert(0, str(ROOT))
    try:
        import cli

        parser = cli.build_parser()
        assert parser.parse_args(["check"]).seeds == 3
        # 仍允许用户显式传小值（运行时由 pipeline 校验拒绝）
        assert parser.parse_args(["check", "--seeds", "2"]).seeds == 2
    finally:
        sys.path.remove(str(ROOT))


@pytest.mark.parametrize("seeds", ["1", "2"])
def test_cli_check_rejects_below_three_seeds(seeds: str) -> None:
    """``--seeds 1/2`` 被pipeline 的 ``n_seeds >= 3`` 约束拒绝（E200）。

    这不是 CLI 的 bug 而是pipeline 的 DoD 约束，但 CLI 未做前置校验，
    导致用户拿到的是裸 traceback 而非友好提示。
    """
    proc = run_cli("check", "--seeds", seeds)
    assert proc.returncode == 1
    assert "E200" in proc.stderr


def test_cli_bench_rejects_methods_flag() -> None:
    """``bench`` **不支持** ``--methods``（argparse rc=2）。

    确认 CLI 的基线筛选只能靠改代码而非命令行参数——这是既定设计
    （``_cmd_bench`` 固定跑全部基线 + 旗舰）。
    """
    proc = run_cli("bench", "--methods", "omp")
    assert proc.returncode == 2
    assert "unrecognized arguments" in proc.stderr
    assert "--methods" in proc.stderr


@pytest.mark.parametrize("cmd", ["ablate", "failures"])
def test_cli_example_subcommands_succeed(cmd: str) -> None:
    """不变式 48b（已修复）：``ablate`` / ``failures`` 子命令现在可用（rc==0）。

    原缺陷：``examples/`` 下只有 ``README.md``，三个子命令 import 的
    ``examples.*`` 模块不存在，全部 ``ModuleNotFoundError`` 失败（rc=1）。
    示例脚本落地后，``ablate`` 打印 JSON、``failures`` 打印复现清单，均可正常运行。
    """
    proc = run_cli(cmd)
    assert proc.returncode == 0, proc.stderr
    if cmd == "failures":
        assert "failure cases reproduced" in proc.stdout
    else:
        assert "full" in proc.stdout  # ablation JSON 含 full 变体


@pytest.mark.slow
def test_cli_demo_writes_benchmark_json(tmp_path: Path) -> None:
    """``demo`` 子命令跑完整基准并把报告写入 ``--out``（rc==0，文件存在）。"""
    out = tmp_path / "benchmark.json"
    proc = run_cli("demo", "--out", str(out), "--seeds", "3")
    assert proc.returncode == 0, proc.stderr
    assert out.exists()
    assert "wrote" in proc.stdout


def test_cli_check_writes_no_files(tmp_path: Path) -> None:
    """``check`` 是纯内存自检，不在项目根目录留下任何产物。"""
    before = {p.name for p in ROOT.iterdir()}
    proc = run_cli("check", "--seeds", "3")
    assert proc.returncode == 0
    after = {p.name for p in ROOT.iterdir()}
    assert after == before, f"check 运行后根目录新增/删除了文件: {after ^ before}"


def test_cli_entrypoint_module_is_importable() -> None:
    """``cli.build_parser`` / ``cli.main`` 可导入（``[project.scripts]`` 入口）。"""
    sys.path.insert(0, str(ROOT))
    try:
        import cli

        assert callable(cli.main)
        parser = cli.build_parser()
        # 默认无子命令时 parse_args 抛 SystemExit(2)
        with pytest.raises(SystemExit) as ei:
            parser.parse_args([])
        assert ei.value.code == 2
    finally:
        sys.path.remove(str(ROOT))
