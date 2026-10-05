# examples

可运行示例脚本目录（v0.1.0 已落地）。

## 可用命令

```bash
python cli.py demo   --seeds 3   # 端到端演示，产出 benchmark.json
python cli.py ablate --seeds 3   # 旗舰四个组件的消融（A1 k估计/A2 路由/A3 去偏/A4 精修）
python cli.py bench  --seeds 3   # 跑基准并打印门禁明细
python cli.py check  --seeds 3   # 确定性自检（bit_identical）
```

`check` 子命令默认 `--seeds 3`（与 pipeline 的 `>= 3 seeds` 约束一致），
无需再显式覆盖。

## 最小示例（无需示例脚本，复制即可运行）

在仓库根目录执行，或先 `pip install -e .`：

```python
from core.config import SolveConfig
from core.seed import set_all
from cs.kselect import noise_discrepancy_k
from cs.solvers import fista, omp
from data.generators import make_instance
from eval.metrics import nmse, support_f1

set_all(20261005)

inst = make_instance("wellcond", n=256, m=96, k=12, snr_db=25.0, seed=0)
print(f"实例 {inst.name}: m={inst.m}, n={inst.n}, k_true={inst.k_true}")

res = omp(inst.A, inst.y, inst.k_true)
print(
    f"OMP   NMSE={nmse(res.x_hat, inst.x_true):.4e}  F1={support_f1(res.x_hat, inst.x_true):.4f}"
)

k_hat, sigma_hat = noise_discrepancy_k(inst.A, inst.y, tau=0.55)
print(f"k 估计: k_true={inst.k_true}, k_hat={k_hat}, sigma_hat={sigma_hat:.5f}")

res2 = fista(inst.A, inst.y, k_hat, SolveConfig(k=k_hat, max_iter=300))
print(
    f"FISTA NMSE={nmse(res2.x_hat, inst.x_true):.4e}  F1={support_f1(res2.x_hat, inst.x_true):.4f}"
)
```

实测输出（Python 3.13.14 + numpy 2.5.3）：

```text
实例 wellcond_snr25: m=96, n=256, k_true=12
OMP   NMSE=6.3670e-04  F1=0.8333
k 估计: k_true=12, k_hat=12, sigma_hat=0.02093
FISTA NMSE=1.7687e-02  F1=0.8000
```

同一段代码也收录在项目 [README](../README.md#最小可运行示例)。

## 约定

每个示例必须是**单文件、可直接运行**的脚本，运行后打印关键指标，
便于作为端到端验证证据。

| 文件 | 对应 CLI 子命令 | 用途 |
| --- | --- | --- |
| `run_demo.py` | `demo` | 端到端演示，产出 `benchmark.json` |
| `ablation.py` | `ablate` | 旗舰四个组件的消融（A1 k估计 / A2 路由 / A3 去偏 / A4 精修） |
| `failures.py` | `failures` | 复现 `references/pitfalls.md` 记录的真实失败/负结果案例 |

作者：晨星
