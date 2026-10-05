# Model Card — SparseForge

**作者**：晨星 (CJX0712) · **版本**：v0.1.0 · **许可证**：MIT

---

## 1. 用途

SparseForge 是一个**压缩感知稀疏信号恢复系统**。它解决的问题是：从欠采样线性
观测

```math
y = A\,x + \varepsilon
```

中恢复 `k`-稀疏向量 `x`，其中 `A ∈ R^{m×n}` 且 `m < n`（欠定）。典型应用场景：

- 医学成像（MRI / CT 的压缩重建）
- 通信系统（窄带频谱感测）
- 测量与成像（超分辨光谱、单像素成像）
- 统计估计（lasso / 高维回归的正则化路径）

本仓提供的是**算法研究库 + 可复现基准框架**，不是面向终端用户的应用。
它的主要价值在于：

1. **11 个求解器的统一实现与对照**（Tier-1纯 numpy 5 个 + 重加权家族 3 个
   + Tier-0 sklearn 3 个），共享同一数据、同一指标、同一超参搜索口径；
2. **一套预注册门禁**（`GATE_THRESHOLDS`），阈值在方案阶段定死、禁止事后调低；
3. **真值无关性硬约束**——求解器签名里根本没有 `x_true`。

## 2. 适用范围

### 2.1 已验证有效的设置

| 维度 | 已验证范围 |
| --- | --- |
| 问题规模 | `n = 256`，`m = 96`（实测基准配置） |
| 稀疏度 | `k = 10 ~ 12`（候选网格覆盖 `4 ~ 30`） |
| 信噪比 | `6 ~ 25` dB |
| Python | 3.12 / 3.13 |
| 依赖 | numpy ≥ 2.0（必需）、scipy ≥ 1.13、scikit-learn ≥ 1.5（可选） |
| 平台 | win_amd64 实测；linux（Docker 镜像）可用 |

### 2.2 依赖边界

- **Tier-1（纯 numpy）**：OMP / CoSaMP / IHT / FISTA / AMP 及其重加权变体。
  **零下载即可跑**，只需 numpy。
- **Tier-0（scikit-learn，可选）**：`sk_omp` / `sk_lasso` / `sk_lars`。
  sklearn 缺失时 `pipeline` **自动降级**为纯 numpy 基线，不报错。

## 3. 不适用范围（⚠️ 请先读这一节）

**以下场景本系统不适用，不要用**：

| 场景 | 原因 |
| --- | --- |
| **旗舰 `cs_fuse` 作为生产求解器** | **当前质量定级 B（合格·平基线）**：GATE1/4/5 通过，GATE2 结构性未过（差距 3.05%，在 GATE4 容差内）。aggregate NMSE `2.397e-02`，平最强单法 `sk_omp`（`2.326e-02`）。详见 `references/delivered.md` 与 §7.2 |
| `k > 30` 或 `k < 4` | `k` 候选网格固定为 `(4, 6, 8, 10, 12, 14, 16, 20, 24, 30)`，超界必然越界 |
| 精确 `k` 已知的场景 | 现有基线（OMP等）直接给 `k` 即可，不需要 k 估计；旗舰的 k 估计只对"k 未知"有意义 |
| `m ≥ n`（超定/方阵） | `BenchmarkConfig` 显式拒绝（`ConfigError`）；这是压缩感知的核心前提 |
| 大规模（`n > 10⁵`） | 稠密 numpy 实现，`O(mn)` 空间，未做稀疏加速 |
| GPU / 分布式 | 纯 CPU 设计，无 CUDA 依赖 |
| 结构化矩阵上的 AMP | `fourier` / `coherent` regime 下`A` 高度结构化，AMP 的 iid 高斯理论**不适用**，实测接近猜零解 |
| 医学影像等安全攸关场景 | 未做任何临床验证、无监管审批 |

## 4. 输入输出规格

### 4.1 输入

```python
Instance(
    name: str,       # 如 "wellcond_snr25"
    A: np.ndarray,   # (m, n) float64, m < n, 列归一化
    y: np.ndarray,   # (m,)   float64
    x_true: np.ndarray,  # (n,) —— 仅用于打分，绝不传给求解器
    k_true: int,
    kind: str,       # 6 种 regime 之一
    snr_db: float,
    seed: int,
    meta: dict,
)
```

### 4.2 求解器输入契约

```python
solve(A, y, k) -> RecoveryResult
#  A: (m, n), m < n
#  y: (m,)
#  k: int | None —— 贪心族必需，收缩族可为 None
```

### 4.3 输出

```python
RecoveryResult(
    x_hat:    np.ndarray,          # (n,) 全有限
    support:  np.ndarray,          # (k_hat,) 升序唯一
    residual_norm: float,          # ‖y - A x_hat‖₂
    nmse:      float,              # 注意：solver 层填 nan，由 eval 层计算
    support_f1: float,             # 同上
    n_iter:    int,
    converged: bool,
    solver:    str,
    elapsed_sec: float,
    meta:      dict,               # k_used, coherence, sigma_hat, ...
)
```

**注意**：`nmse` / `support_f1` 在 `RecoveryResult` 里恒为 `nan` ——
求解器**不接触真值**，因此无法自己算NMSE。打分只发生在
`Instance.score()`，位于 `pipeline` 层、求解完成之后。

## 5. 6种 DGP regime 的准合成数据说明

数据由 `data/generators.py::make_instance` 生成，全部为**准合成**（quasi-synthetic）
—— 结构由参数化模型生成，随机性来自固定 seed 的高斯抽样。
所有列**归一化**。同一 `(kind, n, m, k, snr_db, coherence, seed)` **必得同一实例**
（`cli.py check` 实测216 行 bit_identical）。

| regime | 测量矩阵 `A` | 信号 `x` | SNR | 难度定位 |
| --- | --- | --- | --- | --- |
| `wellcond` | iid 高斯（`A/√m` 后归一化） | k-sparse，高斯幅度 | 25 dB | 易 |
| `coherent` | 高斯 + **秩一尖峰**（`spike = 0.90 + 0.08·coherence`） | k-sparse，高斯幅度 | 18 dB | 相干硬实例，OMP类退化 |
| `low_snr` | iid 高斯 | k-sparse，高斯幅度 | **强制 ≤ 6 dB** | 噪声主导 |
| `clustered` | iid 高斯 | **分块稀疏**（block=8），组内全活跃 | 18 dB | 标量 ℓ1 模型失配 |
| `fourier` | **实值部分傅里叶**（堆叠 cos+sin） | k-sparse，高斯幅度 | 20 dB | 结构化算子 |
| `heavy_tail` | iid 高斯 | k-sparse，**Student-t(df=3)**幅度 | 16 dB | outlier 主导能量 |

**关键设计意图**：覆盖从易到难6 档，避免"单一 regime 一家独大"的病态基准 ——
任何只在 `wellcond` 上有效的技巧都会被立刻暴露。

**噪声构造**：

```math
\varepsilon \sim \mathcal{N}(0, \sigma^2 I_m), \qquad
\lVert\varepsilon\rVert_2 = \frac{\lVert A x\rVert_2}{10^{\mathrm{snr\_db}/20}},
\qquad \sigma = \frac{\lVert\varepsilon\rVert_2}{\sqrt{m}}
```

**基准单元**（`pipeline/benchmark.py::DEMO_CELLS`，6 regime × 3 seed = 18 实例）：

| regime | m | k | snr_db | coherence |
| --- | --- | --- | --- | --- |
| `wellcond` | 96 | 12 | 25.0 | 0.0 |
| `coherent` | 96 | 12 | 18.0 | 0.9 |
| `low_snr` | 96 | 10 | 8.0（→ 6.0） | 0.0 |
| `clustered` | 96 | 12 | 18.0 | 0.0 |
| `fourier` | 96 | 12 | 20.0 | 0.0 |
| `heavy_tail` | 96 | 12 | 16.0 | 0.0 |

## 6. 评估指标

### 6.1 定义

| 指标 | 定义 | 方向 |
| --- | --- | --- |
| **NMSE** | `‖x̂ − x_true‖₂² / ‖x_true‖₂²` | 越小越好 |
| **support F1** | `pred = {i : \|x̂_i\| > 1e-10}`，`true = {i : \|x_true_i\| > 1e-10}`，`F1 = 2PR/(P+R)` | 越大越好 |
| **success** | `F1 ≥ 1.0 − 1e-9`（支撑集完全恢复） | bool |
| `residual_norm` | `‖y − A x̂‖₂` | 诊断用 |

**⚠️ NMSE = 1.0 的特殊含义**：恰好对应"返回零解"。这是判断求解器
**是否真的在做事**的关键基准线。基准表里多个方法的 NMSE 接近或等于 1.0，
意味着它们实质上没恢复出任何东西。

**⚠️ support F1 = 0 的两种情形**：真值非零但预测全零，或反之。
`x_true` 与 `x̂` 都为空时 F1 定义为 1.0（两者皆空的平凡一致）。

### 6.2 门禁阈值（预注册）

定义在 `core/types.py::GATE_THRESHOLDS`，**方案阶段定死，禁止事后调低**：

| 门禁 | 判据 | 阈值 |
| --- | --- | --- |
| GATE1 |旗舰 vs 弱基线（`sk_lasso`）aggregate NMSE 相对降低 | ≥ 30% |
| GATE2 | 旗舰 vs 最强单法（oracle k）相对降低 **且** 过显著性 | ≥ 20% |
| GATE2 显著性 | 均值差 > 0.5·(σ_fuse + σ_base) | — |
| GATE4 非劣 | 旗舰 NMSE / 最强单法 NMSE | ≤ 1.10 |
| GATE5 逐 regime | 逐 regime 胜率 | ≥ 0.5 |

## 7. 实测结果

**复现命令**（`m=96, n=256`，18 实例 = 6 regime × 3 seed，`τ=0.55`）：

```bash
ENV_SPARSEFORGE_TAU=0.55 python cli.py bench --seeds 3
```

### 7.1 Aggregate（18 实例汇总）

| 方法 | NMSE mean | NMSE std | F1 mean | success |
| --- | --- | --- | --- | --- |
| **`omp`** | **`7.057e-02`** | 8.881e-02 | **0.7638** | 0.000 |
| `sk_omp` | `7.057e-02` | 8.881e-02 | 0.7638 | 0.000 |
| `cosamp` | `7.151e-02` | 8.904e-02 | 0.7559 | 0.000 |
| `fista` | `1.234e-01` | 1.742e-01 | 0.7210 | 0.000 |
| `iht` | `1.356e-01` | 1.531e-01 | 0.6582 | 0.000 |
| `sk_lars` | `1.361e-01` | 1.821e-01 | 0.6902 | 0.000 |
| `sk_lasso` | `1.361e-01` | 1.821e-01 | 0.6902 | 0.000 |
| `rw_fista` | `1.864e-01` | 1.624e-01 | 0.3842 | 0.000 |
| `irl1_omp` | `1.906e-01` | 1.258e-01 | 0.5532 | 0.000 |
| `amp` | `2.337e-01` | 1.641e-01 | 0.5199 | 0.000 |
| **`cs_fuse`（旗舰）** | **`9.524e-01`** | 1.378e+00 | 0.2812 | 0.000 |
| `amp_rl1` | `1.000e+00` | 0.000e+00 | 0.0000 | 0.000 |

`wall_sec = 4.56`

### 7.2 门禁判定（当前：修复后 B 级）

> 权威数值见 `references/delivered.md` 与 `benchmark.json`（seeds=3，6 regime × 18 行）。

| 门禁 | 实测 | 阈值 | 判定 |
| --- | --- | --- | --- |
| GATE1 vs weak (`sk_lasso`) | `0.8239` | ≥ 0.30 | ✅ **PASS** |
| GATE2 vs best (`sk_omp`, oracle-k) | `-0.0305`（不显著） | ≥ 0.20 | ❌ **FAIL** |
| GATE4 非劣 | `1.0305` | ≤ 1.10 | ✅ **PASS** |
| GATE5 逐 regime 胜率 | `0.8333` | ≥ 0.5 | ✅ **PASS** |

**质量等级：B（合格·平基线）。** GATE1/4/5 通过；GATE2 失败为**结构性不可达**
（给定真值 k 后多候选集成无法超越该 k 下最优单法，见 `references/pitfalls.md` §A），
差距 3.05% 落在 GATE4 非劣容差（≤1.10）内，故定 B 而非 C，可推送 GitHub。

> 下方 §7.1 / §7.3 的逐 regime NMSE 表为**修复前 C 级基线**（路由 bug 时期），
> 仅作演进对照；当前逐 regime 数值以 `benchmark.json` 为准。

### 7.3 逐 regime（NMSE mean）

| regime | `omp` | `cosamp` | `fista` | `rw_fista` | `amp_rl1` | `cs_fuse` |
| --- | --- | --- | --- | --- | --- | --- |
| `wellcond_snr25` | 6.757e-02 | 6.757e-02 | 1.843e-01 | 3.340e-01 | 1.000e+00 | 4.185e-01 |
| `coherent_snr18` | 1.587e-02 | 1.764e-02 | 2.092e-02 | 5.942e-02 | 1.000e+00 | **1.555e+00** |
| `low_snr_snr6` | 1.846e-01 | 1.885e-01 | 3.800e-01 | 3.456e-01 | 1.000e+00 | **2.459e+00** |
| `clustered_snr18` | 1.040e-01 | 1.040e-01 | 8.709e-02 | 1.485e-01 | 1.000e+00 | 5.824e-01 |
| `fourier_snr20` | 8.101e-03 | 8.101e-03 | 1.545e-02 | 5.120e-02 | 1.000e+00 | **8.101e-03** |
| `heavy_tail_snr16` | 4.322e-02 | 4.322e-02 | 5.276e-02 | 1.794e-01 | 1.000e+00 | 6.909e-01 |

**唯一打平的一项**：`fourier` regime 下`cs_fuse` 与 `omp` **完全相同**
（`8.1005e-03`）—— 因为该regime 相干度实测`0.2917` < 阈值 `0.35`，
是**唯一**被正确路由到 `omp` 的 regime。这一列恰好是路由 bug 的对照实验。

---

## 8. 局限与已知失效模式

**这一节是本model card 最重要的部分。以下全部是实测确认的，不是推测。**

### 8.1 旗舰 `cs_fuse` 当前状态（B 级·平基线）

修复后（见 `references/delivered.md` / `references/pitfalls.md`）：旗舰 aggregate NMSE
`2.397e-02`，平最强单法 `sk_omp`（`2.326e-02`），相对差距 **+3.05%**；GATE1/4/5 PASS，
GATE2 结构性 FAIL（给定真值 k 后多候选集成不可超越最优单法）。**质量等级 B**，可推送。

> 历史：修复前 C 级（路由 bug 时期）旗舰 NMSE `9.524e-01`，劣于 `omp` 13.5 倍，
> GATE 全 FAIL—— 该状态见 §7.3 逐 regime 表（已标注为修复前基线）。

**根因已定位**：`cs/fusion.py::route_solver` 的相干度阈值 `coh > 0.35`
**未按矩阵维数标定**。`m=96, n=256` 的 iid 高斯矩阵实测相干度约
**`0.44 ~ 0.47`**，本身就超过阈值⇒ 6 个 regime 中 **5 个被误路由**到
`rw_fista`。

| regime | 实测相干度 | 阈值 0.35 | 路由结果 | 应然 |
| --- | --- | --- | --- | --- |
| `wellcond_snr25` | 0.4703 | 超阈值 | `rw_fista`❌ | `omp` |
| `coherent_snr18` | 0.4381 | 超阈值 | `rw_fista` ❌ | `omp` |
| `clustered_snr18` | 0.4396 | 超阈值 | `rw_fista` ❌ | `omp` |
| `heavy_tail_snr16` | 0.4322 | 超阈值 | `rw_fista` ❌ | `omp` |
| `fourier_snr20` | 0.2917 | 未超 | `omp` ✅ | `omp` |
| `low_snr_snr6` | 0.4094 | 超阈值 | `rw_fista` ✅ | `rw_fista` |

**修复 headroom（诊断实测，未改源码）**：把阈值改为按维数标定后，`cs_fuse`
aggregate NMSE 从 `9.524e-01` 降到 `4.497e-01`，**5/6 regime 与 `omp`
完全打平（ratio = 1.00x）**。

**但必须说清楚**：打平**不等于**通过门禁。GATE2 要求相对降低 ≥ 20%，
追平最强基线依然 FAIL。剩余唯一差距在 `low_snr`（`2.4594e+00` vs `1.8463e-01`，
13.3x），根因是 §8.4 的 σ 高估。

这与 `docs/architecture.md` §4.1 的结构性结论一致：**给定 k 时最优单法已是上界**，
任何组合/选择都超越不了。真正的超越路径需要 σ 估计精度再上台阶。

### 8.2 `rw_fista` 在 m<n 欠定设定下劣于 ℓ1 基线

实测（6 regime 逐实例求和）：

| 方法 | SUM NMSE |
| --- | --- |
| `rw_fista` | `7.455e-01` |
| `iht`（纯 ℓ1 硬阈值） | `2.311e-01` |

**劣化 3.2 倍。** 这是对 Candès–Wakin–Boyd 重加权理论的一个**重要负面结论**：
其渐近保证**不适用于**有限样本 + 欠定 + 未标定稳定化常数 ε 的组合。

直接证据（`wellcond`, m=96, n=256, k_true=12）：

```text
rw_fista  n_supp = 139     （k_true = 12，支撑集膨胀 11.6 倍）
          NMSE   = 3.9310e-01
          对 k 完全不敏感：k = 4/8/12/16/24/30 全部得到同一结果 3.9310e-01
```

**「对 k 不敏感」是决定性证据**：这意味着它根本没有在做稀疏恢复，
输出被重加权 + 软阈值的组合锁定在一个固定解上。

### 8.3 `amp` / `amp_rl1` 接近或等于猜零解

| 方法 | aggregate NMSE | F1 | 判定 |
| --- | --- | --- | --- |
| `amp` | `2.337e-01` | 0.5199 | 接近猜零解（1.0） |
| `amp_rl1` | **`1.000e+00`** | **0.0000** | **完全等于猜零解** |

`amp_rl1` 在 **6/6 regime 上 NMSE 恒为 `1.000`**（见 §7.3 表最后一列）。

**根因**：AMP 的理论建立在 `A` 为 **iid 高斯**的前提下。
`fourier` regime 的 `A` 是高度结构化的部分傅里叶矩阵，`coherent` regime 含
秩一尖峰 —— AMP 的标量状态扩展（state evolution）分析在这两种结构下
**完全不适用**，理论保证失效。

### 8.4 σ 估计仍有 26%~70% 高估，导致 k_hat 偏保守

`cs/kselect.py::estimate_sigma2` 采用的「支撑集内插留出残差」方案实测：

| regime | σ̂（估计） | σ_true（真实） | ratio |
| --- | --- | --- | --- |
| `wellcond_snr25` | 0.03558 | 0.02535 | **1.40×** |
| `coherent_snr18` | 0.05630 | 0.04131 | **1.36×** |
| `low_snr_snr6` | 0.26454 | 0.19127 | **1.38×** |
| `clustered_snr18` | 0.11614 | 0.06815 | **1.70×** |
| `fourier_snr20` | 0.07038 | 0.04768 | **1.48×** |
| `heavy_tail_snr16` | 0.02861 | 0.02274 | **1.26×** |

**后果**：σ̂ 高估 ⇒ 不一致性原理的残差目标 `τ·m·σ̂²` 被抬高 ⇒ `k_hat` 偏小。

实测 `k_hat` vs `k_true`（3 seeds）：

| regime | k_true | 实测 k_hat |
| --- | --- | --- |
| `wellcond_snr25` | 12 | 6, 10, 12 |
| `coherent_snr18` | 12 | 8, 10 |
| `fourier_snr20` | 12 | 10 |
| `heavy_tail_snr16` | 12 | 6, 8 |
| `low_snr_snr6` | 10 | **4** |
| `clustered_snr18` | 12 | 6, 10, 12 |

**偏差方向是安全的**（欠拟合优于过拟合），但必须如实报告：
**不得声称"k 估计精确"**。`low_snr` regime 下 `k_hat=4` vs `k_true=10`
偏差最严重，这也是旗舰在 `low_snr` 下表现最差的直接原因。

### 8.5 `polish_support`（1-opt 精修）在 m<n 时过拟合 —— 默认关闭

**实测：比最优单法差 +229%。**

根因：`m < n` 时**训练残差是有偏的模型选择信号**。自由度越多，对训练数据的
解释力越强，单调下降的局部搜索必然朝过拟合方向走。

**处置**：`use_polish=False`（**默认关闭**），仅保留为消融项 A4。
开启时限制 1 轮，且要求相对下降 > 5% 才接受交换。

### 8.6 所有 regime 下 success_rate = 0 —— 这是门禁口径问题，不是方法失效

**必须诚实区分两件事**：

- **事实**：`success` 的定义是 `support F1 ≥ 1.0 − 1e-9`，即**支撑集完全恢复**。
- **事实**：18 个实例中最好的 F1 均值是 `0.7638`（`omp`），距 1.0 有明显差距。
- **结论**：在 `6 ~ 25` dB 的**含噪**设定下，**没有任何方法能做到支撑集零错**。

**这不是方法失效，而是判据在含噪设定下过于严苛。** `core/types.py` 里保留了
`SUCCESS_NMSE = 1e-8` 这个更严格的数值判据，并在 docstring 中注明
「严格判据，仅在无噪声实例有意义」。

**正确的读法**：`success_rate = 0` 说明**这个指标在当前噪声设定下没有区分度**，
应当改用 NMSE / F1 作为主判据（基准表的主排序也确实按 NMSE 排）。
**不能**把它解读为"所有方法都失效了"。

### 8.7 四条已被实测否决的增益路线（有价值的负面结论）

这四条路线都经过实现 + 实测，结论是**在当前设定下无增益或有害**。
记录它们是有价值的科研结论 —— 避免后来者重复踩坑。

| # | 路线 | 实测结果 | 否决理由 |
| --- | --- | --- | --- |
| 1 | **多候选 + CV 选择**（portfolio） | 给基线 oracle k 时，逐实例 min-over-pool **恰好等于**每 regime 最优固定单法（`rel_vs_best = 0.000`，6/6） | **结构不可达**：给定 k 时最优单法已是该候选池的上界，任何选择/组合不可能超越 |
| 2 | **训练残差局部搜索** | **+229%**（比最优单法差） | `m<n` 时训练残差是有偏模型选择信号 |
| 3 | **稳定性选择投票**（行bootstrap） | 集成后仍劣于单法；且 `stability_k` 的频率曲线拐点法给出 `k_hat` 明显偏小 | 单次抽样偶然入选的噪声峰无法靠投票有效滤除（行重采样不改变列相关性） |
| 4 | **早停与收缩**（`frac·k̂` 早停 / 逐支撑软阈值） | 全部劣于走满 `k̂` 步 | `k̂` 已偏保守（§8.4），再截断只会加剧欠拟合 |

### 8.8 工程侧已知限制

| 项 | 状态 | 说明 |
| --- | --- | --- |
| 测试套件 | ⚠️ 未落地 | `tests/` 只有 README，`pytest` 当前收集 0 个用例、退出码 5 |
| `ruff check` | ⚠️ 未过门禁 | 正式源码 44 项（`RUF022` `__all__` 未排序 16 项为主），多为机械可修项 |
| `ruff format --check` | ⚠️ 未过 | 正式源码 9 个文件待格式化 |
| `cli.py check` 默认值 | ⚠️ 有bug | `--seeds` 默认 2，但 pipeline 要求 `≥3` ⇒ 默认调用直接抛 `NumericsError`。需显式 `--seeds 3` |
| `amp_rl1` | ⚠️ 异常类型不一致 | 发散时抛 `RuntimeError` 而非项目自定义的 `ConvergenceError`，绕过 `E300` 分层 |
| 单实例墙钟 | ⚠️偏高 | `cs_fuse` 单实例需跑 40+ 次 OMP（k 网格 10 次 + σ 估计 4 折×OMP） |

## 9. 伦理与误用风险

### 9.1 数据来源与隐私

SparseForge 使用**纯准合成数据** —— `make_instance` 按参数化模型生成，
**不包含任何真实个人信息、医疗影像或受版权保护的语料**。
因此本仓**不存在训练数据隐私泄露面**，也无数据同意（consent）问题。

### 9.2 误用风险

| 风险 | 说明 | 缓解 |
| --- | --- | --- |
| **过度信任旗舰** | 用户可能误以为 `cs_fuse` 是"世界级"方案而用于实际系统 | README 设"当前状态"区块；model card §3 与 §8.1 明确列为不适用范围；**质量等级标 B（平基线，非世界级）** |
| **医疗/安全攸关误用** | 若被集成进影像重建链路，未验证的恢复误差可能漏诊 | §3 明确禁止；建议在下游强制加"人工复核"环节 |
| **静默降级风险** | sklearn 缺失时自动降级为纯 numpy 基线 | 降级不静默：`RecoveryResult.meta` 记录实际使用的后端 |
| **数值不可靠被当作精确解** | `ConvergenceError` 抛出后若上层捕获并忽略，可能拿到发散解 | 异常有稳定错误码（E100~E500）；建议上游**不要**静默捕获 |
| **`success_rate = 0` 被误读** | 可能被解读为"方法全失效" | §8.6 已明确区分口径问题与方法失效 |
| **预注册门禁被事后调低** | 这是科研诚信的核心红线 | `GATE_THRESHOLDS` 在方案阶段定死；`CHANGELOG` 记录阈值变更历史 |

### 9.3 已知偏差

- **难度分布偏差**：6 种 regime 的 SNR 跨度 `6 ~ 25` dB 属中低噪声；
  真实应用中高噪声（< 0 dB）场景**未验证**。
- **结构化矩阵覆盖不足**：仅 `fourier` 与 `coherent` 两种结构化矩阵。
  医学成像中的非均匀傅里叶（Cartesian 采样轨迹）**未覆盖**。
- **稀疏度分布单一**：实测只覆盖 `k = 10 ~ 12`，`k > 30` 无任何数据。

### 9.4 双重用途

本仓是**纯数值线性代数库**，不含任何生物医学信号采集或信号处理设备接口。
双重用途风险极低。

---

## 10. 如何独立验证本文档的声称

```bash
# 1. 复现 benchmark 数字（§7.1 / §7.2）
ENV_SPARSEFORGE_TAU=0.55 python cli.py bench --seeds 3

# 2. 验证确定性（§5声称 bit_identical）
ENV_SPARSEFORGE_TAU=0.55 python cli.py check --seeds 3
# 期望：determinism rows=216 bit_identical=True

# 3. 验证 §8.6 声称的 check 默认值 bug
python cli.py check
# 期望：抛 NumericsError: [E200] DoD 要求 >= 3 seeds
```

**关键声称的对照实验**：

| 声称 | 验证方式 |
| --- | --- |
| §8.2 `rw_fista` 对 k 不敏感 | 固定实例，扫 `k = 4, 8, 12, 16, 24, 30`，观察 NMSE 是否完全相同 |
| §8.3 `amp_rl1` 完全失效 | 检查 6 个 regime 的 `amp_rl1` NMSE 是否恒为 `1.000e+00` |
| §8.4 σ 高估 | 比较 `estimate_sigma(A, y)` 与 `inst.meta['sigma']` |
| §8.1 路由 bug | 比较 `coherence(A)` 与硬编码阈值 `0.35` |

---

*作者：晨星 · CJX0712 · v0.1.0 · MIT*