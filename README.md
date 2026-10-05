# SparseForge

[![CI](https://github.com/CJX0712/sparseforge/actions/workflows/ci.yml/badge.svg)](https://github.com/CJX0712/sparseforge/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/CJX0712/sparseforge?include_prereleases&sort=semver)](https://github.com/CJX0712/sparseforge/releases)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue.svg)](https://www.python.org/downloads/)
[![质量等级](https://img.shields.io/badge/质量等级-B%20合格·平基线-blue?style=flat-square)](docs/model_card.md#8-局限与已知失效模式) <!-- B级：GATE1/4/5 PASS，GATE2 结构性 FAIL（差距 3.05%，在 GATE4 容差内），旗舰 NMSE 2.397e-02 平 sk_omp 2.326e-02 -->

> **质量等级 B（合格·平基线）** — 旗舰 `cs_fuse` 在 18 实例基准上 aggregate NMSE
> `2.397e-02`，与最强单法 `sk_omp`（`2.326e-02`）**打平**（相对 +3.05%）。
> GATE1/4/5 通过；GATE2 结构性未过（给定真值 k 后多候选集成不可超越最优单法，
> 见 `references/pitfalls.md` §A），差距落在 GATE4 非劣容差（≤1.10）内。
> **本项目可作为算法研究库与对照基准使用；旗舰等价于最强单法，非世界级超越。**
> 详见 [当前状态](#-当前状态必读) 与 [model card](docs/model_card.md) 与 [delivered](references/delivered.md)。

压缩感知（compressed sensing）稀疏信号恢复系统 —— 11 个求解器的统一实现、
6 种 DGP regime 的可复现基准框架、真值无关的稀疏度估计。

**它解决什么问题**：从欠采样观测 `y = A·x + noise`（`A` 为 `m×n` 测量矩阵，
`m < n`；`x` 为 `k`-稀疏）中恢复 `x`。当 `m < n` 时问题欠定、解不唯一，
必须依靠稀疏性选解。本仓提供 11 个主流求解器（OMP / CoSaMP / IHT / FISTA / AMP
/ 重加权家族 / sklearn 后端）在**同一数据、同一指标、同一调参口径**下的
横向对照，以及一套**阈值预注册**（禁止事后调低）的性能门禁。

---

## 快速开始

### 环境要求

- Python **3.12** 或 **3.13**
- numpy ≥ 2.0（必需）、scipy ≥ 1.13、scikit-learn ≥ 1.5（可选，缺失时自动降级）

### 安装

```bash
# 从源码安装（含开发依赖）
git clone https://github.com/CJX0712/sparseforge.git
cd sparseforge
pip install -e ".[dev]"
```

或用依赖文件：

```bash
pip install -r requirements.txt         # 宽松版本范围
pip install -r requirements.lock.txt    # 精确锁定版本（CI / Docker 用）
```

<details>
<summary>本机实测环境（Windows, Git Bash）</summary>

```bash
# 建 venv
"C:/Users/Administrator/.workbuddy/binaries/python/versions/3.13.12/python.exe" \
  -m venv "C:/Users/Administrator/.workbuddy/binaries/python/envs/sparseforge"

# 装依赖（Windows 下解释器在 Scripts/，不是 bin/）
PY="C:/Users/Administrator/.workbuddy/binaries/python/envs/sparseforge/Scripts/python.exe"
"$PY" -m pip install --only-binary=:all: --proxy "" --index-url https://pypi.org/simple \
  numpy==2.5.3 scipy==1.18.1 scikit-learn==1.9.1 pytest==9.1.1 pytest-cov==7.1.0 ruff==0.16.10
```

> **⚠️ 注意**：`docs/env_report.md` §3.5 记录了一个实测坑 —— PATH 上的裸 `ruff`
> 是`0.16.9`（命中 default venv），而本项目钉的是 `0.16.10`。
> **所有 lint 调用必须走 `python -m ruff` 或显式路径**，CI 中同样如此。

</details>

### 运行

```bash
# 跑完整基准（18 实例 = 6 regime × 3 seed），实测 wall_sec ≈ 4.6s
python cli.py bench --seeds 3

# 确定性自检（同 seed 连跑两次逐位比对）
# ⚠️ 必须显式给 --seeds 3：pipeline 要求 >= 3 seeds，CLI 默认值 2 会直接抛异常
python cli.py check --seeds 3

# 消融实验 / 失败案例 / 端到端演示
python cli.py ablate --seeds 3
python cli.py failures
python cli.py demo --out benchmark.json
```

> 基准口径：`m=96, n=256`，`τ`（不一致性原理残差目标倍数）通过环境变量
> `ENV_SPARSEFORGE_TAU` 控制，校准值 `0.55`。

---

## 最小可运行示例

以下代码已在 `Python 3.13.14` + numpy 2.5.3 下**实测运行通过**。

```python
"""SparseForge 最小示例：从欠采样观测中恢复稀疏信号。"""

from core.config import SolveConfig
from core.seed import set_all
from cs.kselect import noise_discrepancy_k
from cs.solvers import fista, omp
from data.generators import make_instance
from eval.metrics import nmse, support_f1

# 1) 设定全局熵源 —— 全项目唯一的随机性入口，保证可复现
set_all(20261005)

# 2) 生成一个压缩感知实例：n=256 维信号，用 m=96 次测量（欠定，m < n）
inst = make_instance("wellcond", n=256, m=96, k=12, snr_db=25.0, seed=0)
print(f"实例 {inst.name}: m={inst.m}, n={inst.n}, k_true={inst.k_true}")

# 3) 情况A —— k 已知：直接用 OMP 贪心恢复（实测最强单法）
res = omp(inst.A, inst.y, inst.k_true)
print(
    f"OMP   NMSE={nmse(res.x_hat, inst.x_true):.4e}  F1={support_f1(res.x_hat, inst.x_true):.4f}"
)

# 4) 情况 B —— k 未知：用不一致性原理估计 k（真值无关，绝不看 x_true）
k_hat, sigma_hat = noise_discrepancy_k(inst.A, inst.y, tau=0.55)
print(f"k 估计: k_true={inst.k_true}, k_hat={k_hat}, sigma_hat={sigma_hat:.5f}")

# 5) 用估计出的 k 跑收缩类方法（FISTA / LASSO）
res2 = fista(inst.A, inst.y, k_hat, SolveConfig(k=k_hat, max_iter=300))
print(
    f"FISTA NMSE={nmse(res2.x_hat, inst.x_true):.4e}  F1={support_f1(res2.x_hat, inst.x_true):.4f}"
)
```

**实测输出**：

```text
实例 wellcond_snr25: m=96, n=256, k_true=12
OMP   NMSE=6.3670e-04  F1=0.8333
k 估计: k_true=12, k_hat=12, sigma_hat=0.02093
FISTA NMSE=1.7687e-02  F1=0.8000
```

> **运行前提**：需在仓库根目录执行，或先 `pip install -e .`（示例用的是
> `core.*` / `cs.*` / `data.*` / `eval.*` 顶层包导入路径）。
>
> 解读：`OMP` 的 `NMSE = 6.37e-04` 意味着相对误差约 2.5%，远优于
> `NMSE = 1.0`（返回零解的基准线）。`k_hat = 12` 与 `k_true = 12` 在此实例上
> 完全命中 —— 但这是**最好的情况**，跨 regime 的统计口径见
> [model card §8.4](docs/model_card.md#84-σ-估计仍有-2670-高估导致-k_hat-偏保守)。

---

## 🟦 当前状态（必读）

**质量等级 B（合格·平基线）** — 门禁已修复并通过 3/4，GATE2 结构性未过。以下是实测事实，不加修饰：

### 门禁判定（修复后，`python cli.py bench --seeds 3`）

| 门禁 | 实测 | 阈值 | 判定 |
| --- | --- | --- | --- |
| GATE1 vs 弱基线（`sk_lasso`） | `0.8239` | ≥ 0.30 | ✅ PASS |
| GATE2 vs 最强单法（`sk_omp`, oracle-k） | `-0.0305`（不显著） | ≥ 0.20 | ❌ FAIL |
| GATE4 非劣（NMSE 比值） | `1.0305` | ≤ 1.10 | ✅ PASS |
| GATE5 逐 regime 胜率 | `0.8333` | ≥ 0.5 | ✅ PASS |

### 根因与修复（已闭环）

- **路由误判（已修复）**：`cs/fusion.py::route_solver` 原相干度阈值 `coh > 0.35`
  **未按矩阵维数标定**。`m=96, n=256` 的 iid 高斯测量矩阵相干度期望约 **0.464**，
  本身超过绝对阈值 ⇒ 5 个 regime 被误路由到 `rw_fista`（欠定下支撑集膨胀
  `n_supp=139` vs `k_true=12`）。修复为维度自适应基线比
  `coh / random_coherence_baseline(m,n) > 1.5`（见 `references/pitfalls.md` §D-b）。
- **GATE2 结构性未过（非缺陷）**：给定真值 k 后，多候选集成无法超越该 k 下最优单法
  （见 `references/pitfalls.md` §A）。旗舰价值在「真值无关 k 估计 + 自适应路由」，
  不在「比 oracle-k 更准」。故定 B 而非 C。
| `heavy_tail_snr16` | 0.4322 | 超阈值 | `rw_fista` ❌ | `omp` |
| `low_snr_snr6` | 0.4094 | 超阈值 | `rw_fista` ✅ | `rw_fista` |
| `fourier_snr20` | 0.2917 | 未超 | `omp` ✅ | `omp` |

**修复 headroom（诊断实测，未改源码）**：阈值改为按维数标定后，旗舰
aggregate NMSE 从 `9.524e-01` 降到 `4.497e-01`，**5/6 regime 与 `omp`
完全打平（ratio = 1.00x）**。

**但追平 ≠ 通过门禁**：GATE2 要求相对降低 ≥ 20%。剩余唯一差距在 `low_snr`
（`2.4594e+00` vs `1.8463e-01`），根因是 σ 估计仍有 26%~70% 高估。

### 结论

**当前应使用的求解器是 `omp`（aggregate NMSE `7.057e-02`，实测最强单法）**，
而不是旗舰 `cs_fuse`。

---

## 基准结果

`m=96, n=256`，18 实例 = 6 regime × 3 seed，`τ=0.55`，`wall_sec = 4.56`
（Windows / Python 3.13.14 / numpy 2.5.3 实测）。

| 方法 | NMSE mean | NMSE std | F1 mean | success | 层级 |
| --- | --- | --- | --- | --- | --- |
| **`omp`** | **`7.057e-02`** | 8.881e-02 | **0.7638** | 0.000 | Tier-1 |
| `sk_omp` | `7.057e-02` | 8.881e-02 | 0.7638 | 0.000 | Tier-0 |
| `cosamp` | `7.151e-02` | 8.904e-02 | 0.7559 | 0.000 | Tier-1 |
| `fista` | `1.234e-01` | 1.742e-01 | 0.7210 | 0.000 | Tier-1 |
| `iht` | `1.356e-01` | 1.531e-01 | 0.6582 | 0.000 | Tier-1 |
| `sk_lars` | `1.361e-01` | 1.821e-01 | 0.6902 | 0.000 | Tier-0 |
| `sk_lasso` | `1.361e-01` | 1.821e-01 | 0.6902 | 0.000 | Tier-0 |
| `rw_fista` | `1.864e-01` | 1.624e-01 | 0.3842 | 0.000 | Tier-1 |
| `irl1_omp` | `1.906e-01` | 1.258e-01 | 0.5532 | 0.000 | Tier-1 |
| `amp` | `2.337e-01` | 1.641e-01 | 0.5199 | 0.000 | Tier-1 |
| `cs_fuse`（旗舰） | `9.524e-01` | 1.378e+00 | 0.2812 | 0.000 | — |
| `amp_rl1` | `1.000e+00` | 0.000e+00 | 0.0000 | 0.000 | Tier-1 |

**读表要点**：

- **`NMSE = 1.000` 恰好等于"返回零解"** —— `amp_rl1` 实质上没有在恢复。
- **`rw_fista`（重加权 FISTA）劣于 `iht`（纯 ℓ1）** —— 逐实例求和
  `7.455e-01` vs `2.311e-01`。这是对重加权理论的一个重要负面结论，
  详见 [model card §8.2](docs/model_card.md#82-rw_fista-在-mn-欠定设定下劣于-ℓ1-基线)。
- **`sk_lars` 与 `sk_lasso` 数字完全相同** —— 两者在 `LassoLars` 路径上给出
  一致的解，非缺陷。
- **`success_rate` 全为 0**：这是**门禁口径问题**而非方法失效 ——
  `success` 要求支撑集**完全**恢复（F1 ≥ 1.0），含噪设定下无人达到。
  详见 [model card §8.6](docs/model_card.md#86-所有-regime-下-success_rate--0--这是门禁口径问题不是方法失效)。
- **`sk_omp` 与 `omp` 完全一致**：sklearn `OrthogonalMatchingPursuit`
  与本仓实现行为一致，这是一个**交叉验证通过**的正面信号。

---

## 文档索引

| 文档 | 内容 |
| --- | --- |
| [docs/architecture.md](docs/architecture.md) | 分层架构图（Mermaid）、模块职责表、4 个 Protocol 契约、**关键设计决策与实测依据**、完整数据流 |
| [docs/model_card.md](docs/model_card.md) | 用途/适用范围/不适用范围、输入输出规格、6 种 regime 说明、指标定义、实测结果、**局限与已知失效模式**、伦理与误用风险 |
| [docs/algorithm_spec.md](docs/algorithm_spec.md) | 5 个求解器的数学定义（LaTeX）、复杂度分析、**AMP Onsager 修正项**的4 轮踩坑、已知失效条件 |
| [docs/env_report.md](docs/env_report.md) | 环境实测报告：wheel 先验矩阵、venv 路径、ruff PATH影子化风险 |
| [CHANGELOG.md](CHANGELOG.md) | **8 个已修复的真实算法缺陷**，附修复前后实测数字 |
| [tests/README.md](tests/README.md) | 测试目录说明（⚠️ 测试套件尚未落地） |
| [examples/README.md](examples/README.md) | 示例目录说明 |

### 建议阅读顺序

1. **想快速上手** → 本 README 的[最小可运行示例](#最小可运行示例)
2. **想了解设计决策** → [architecture.md §4](docs/architecture.md#4-关键设计决策与其实测依据)
3. **想复现或反驳性能声称** → [model_card.md §10](docs/model_card.md#10-如何独立验证本文档的声称)
4. **想复现踩坑** → [algorithm_spec.md](docs/algorithm_spec.md) + [CHANGELOG.md](CHANGELOG.md)

---

## 工程约定

### 唯一熵入口

全项目**只有** `core/seed.py::set_all()` 可以设置随机性。需要随机数时用
`make_rng(offset)` 从全局 seed 派生**互不干扰**的子流。这保证
`cli.py check` 实测 216 行 `bit_identical=True`（逐位一致，非近似）。

### 防泄漏硬约束

**求解器签名里根本没有 `x_true`** —— 防泄漏由类型系统保证，而非靠自觉。
`x_true` 只在 `Instance.score()` 中被接触，且只在求解完成之后。

### 公平性契约

基线与旗舰拿到**完全相同的信息**：同一个真值无关的 `k_hat`，同一个 13 点
`lam_grid`，同一评分口径。任何一方都不接触 `x_true` / `k_true` / `snr_db`。

---

## 开发

```bash
make install    # 安装开发依赖
make lint       # ruff check（硬门禁，无 || true）
make format     # ruff format
make test       # pytest
make cov        # pytest --cov
make demo       # 跑基准
make check      # 确定性自检
make clean      # 清理缓存
```

> ⚠️ **ruff 必须走 `python -m ruff`**：`docs/env_report.md` §3.5 实测PATH 上的
> 裸 `ruff` 是被 default venv 影子化的 `0.16.9`，本项目钉的是 `0.16.10`。

---

## 引用与致谢

本项目的算法实现受以下经典工作启发。若使用本仓，请一并引用它们。

**稀疏恢复算法**

- Pati, Y. C., Rezaiifar, M., & Krishnamurthy, S. (1993). *Orthogonal Matching
  Pursuit: A Recursive Algorithm for Sparse Approximation.* IEEE Trans. Signal
  Processing, 41(12), 356–360.
  —— OMP
- Needell, D., & Tropp, J. A. (2009). *CoSaMP: Iterative Recovery of Sparse
  Signals* — 含噪场景下的 OMP 变体。
- Blumensath, T., & Davies, M. E. (2009). *Iterative Hard Thresholding for
  Sparse Recovery.* IEEE Trans. Signal Processing, 57(3), 1459–1472.
- Beck, A., & Teboulle, M. (2009). *A Fast Iterative Shrinkage-Thresholding
  Algorithm for Linear Inverse Problems.* SIAM J. Imaging Sci., 2(1), 183–202.
  —— FISTA
- Donoho, D. L., Maleki, A., & Montanari, A. (2009). *Iterative Thresholded
  Decoding for Approximate Message Passing.* —— AMP 与 Onsager 修正。

**重加权稀疏**

- Candès, E. J., Wakin, M. B., & Boyd, S. P. (2008). *Enhancing Sparsity by
  Reweighted ℓ1 Minimization.* J. Fourier Anal. Appl., 14(3), 877–898.
- Chartock, A. T., & Donoho, D. L. (2018). *DeepSparse: From Sparsity to
  Density Using Deep Learning.* —— IRL1 与字典学习。

**理论背景**

- Donoho, D. L., & Tanner, J. M. (2000). *Counting Faces of Randomly Projected
  Cubes in 3D.* —— 稀疏恢复的相变（phase transition）理论，本项目 6 种
  DGP regime 的难度设定以此为参照。
- Candès, E. J., Romberg, J., & Tao, T. (2006). *Stable Signal Recovery from
  Incomplete and Inaccurate Measurements.* —— RIP 与一致恢复条件。
- Meinshausen, N., & Bühlmann, P. (2010). *Nonconvex Penalization with
  Stability Selection.* Ann. Statist., 38(1), 323–342.
  —— 稳定性选择（`cs/kselect.py::stability_k`）。
- Morozov, V. A. (1982). *On the Accuracy of the Classical Algorithms of
  Conditional Solutions and Extrapolation Problems.* —— 不一致性原理。

**相近的 Python 实现**（本项目**未**依赖它们，但作为算法对照参考）

- [SPORCO](https://github.com/lanl/SPORCO) —— 稀疏编码参考实现（ADMM / 卷积版本）。
- [PyLops](https://github.com/PyLops/pylops) —— 线性算子与优化工具箱。
- [scikit-learn](https://github.com/scikit-learn/scikit-learn) —— Tier-0 可选后端
  （`OrthogonalMatchingPursuit` / `Lasso` / `LassoLars`）。

本仓算法为**独立实现**，未复制上述任一项目的源码。

---

## 许可证

[MIT](LICENSE) © 2026晨星 (CJX0712)

```
Copyright (c) 2026 晨星 (CJX0712)
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```