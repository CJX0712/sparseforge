# SparseForge 算法规格（数学定义）

> 作者：晨星 · CJX0712
> 适用范围：`v0.1.0`
> 约定：`A ∈ R^{m×n}`，`m < n`（欠定）；`x ∈ R^n` 为 `k`-稀疏；`y ∈ R^m` 为观测。
> 全文复杂度以稠密 numpy 实现为准，未使用稀疏结构加速。

---

## 0. 问题设定

稀疏恢复（压缩感知）要求从线性观测

```math
y = A\,x + \varepsilon,\qquad \varepsilon \sim \mathcal{N}(0, \sigma^2 I_m)
```

中恢复 `k`-稀疏向量 `x`（`k ≪ n`）。当 `m < n` 时问题**欠定**，解不唯一，
必须依靠稀疏性假设选解。

**评测指标**（`eval/metrics.py`，与 `core/types.py` 语义一致）：

- **NMSE**（越小越好）

  ```math
  \mathrm{NMSE} = \frac{\lVert \hat{x} - x_{\mathrm{true}}\rVert_2^2}{\lVert x_{\mathrm{true}}\rVert_2^2}
  ```

  `NMSE = 1.0` 恰好对应"返回零解"，是判断求解器是否**真的在做事**的基准线。

- **支撑集 F1**（越大越好）：取 `pred = {i : |x_i| > ε}`（`ε = 1e-10`）、
  `true = {i : |x_true_i| > ε}`，则

  ```math
  \mathrm{F1} = \frac{2\,\mathrm{precision}\cdot\mathrm{recall}}{\mathrm{precision} + \mathrm{recall}}
  ```

- **success**：定义为 `F1 ≥ 1.0 - 1e-9`，即**支撑集完全恢复**。

**防泄漏硬约束**：所有求解器的输入只有 `(A, y, k)`。`x_true` / `k_true` /
`snr_db` **绝不传入** `solve()`。这不是自觉，而是由函数签名保证——
`x_true` 在类型上就不可达（见 `docs/architecture.md` §4.2）。

---

## 1. OMP — Orthogonal Matching Pursuit

> Pati, Rezaiifar & Krishnamurthy (1993), *Orthogonal Matching Pursuit* —
> 贪心族基线，本项目实测最强单法。

### 更新式

```math
S_0 = \varnothing
\quad
S_{j+1} = S_j \cup \left\{ i^* = \arg\max_{i \notin S_j} \left| \langle a_i,\, r_j \rangle \right| \right\}
\quad
x_{j+1} = \arg\min_{x:\,\mathrm{supp}(x) = S_{j+1}} \lVert y - A x \rVert_2
\quad
r_{j+1} = y - A x_{j+1}
```

每轮：算相关 `|Aᵀr|` → 排除已选列 → 取最大者并入支撑集 → 在支撑集上
**最小二乘重拟合** → 重算残差。实现见 `cs/solvers.py::omp`。

**超参**：`k`（必需，贪心族必须有稀疏度）、`max_iter`、`tol`。

**收敛条件**：跑满 `k` 轮即停（`converged = (len(sup) == k)`）。
另有提前退出：若 `corr[i] <= 0`（残差已无相关成分）则break。

### 复杂度

每轮一次 `Aᵀr`（`O(mn)`）+ 支撑集上最小二乘（`O(mk + k³)`）。
总**时间 `O(k·mn)`**，**空间 `O(mn)`**（稠密存A）。瓶颈在 `Aᵀr`，
可增量化为 `O(k·mn)` →但本实现未优化。

### 已知失效条件

- **相干矩阵**：若两列近乎共线（`coherence → 1`），贪心选列失去区分度。
  实测 `coherent` regime 下 `fourier` 类退化。
- **`k` 严重高估**：若 `k_hat ≫ k_true`，支撑集被噪声峰填满（实测
  `OMP 残差 MAD` 因此低估 σ，见 §5）。
- **`k` 严重低估**：支撑集不够，解欠拟合。

---

## 2. CoSaMP — Compressive Sampling Matching Pursuit

> Needell & Tropp (2009), *CoSaMP: Iterative Recovery of Sparse Signals*。

### 更新式

```math
\Omega \leftarrow \text{top-}2k \text{ of } \left| A^\top r \right|
\quad
\Omega \leftarrow \Omega \cup S
\quad
\tilde{x} \leftarrow \arg\min_{\mathrm{supp}(x) \subseteq \Omega} \lVert y - A x \rVert_2
\quad
S \leftarrow \text{top-}k \text{ of } \left| \tilde{x} \right| \;\text{(按幅值)}
\quad
x \leftarrow \arg\min_{\mathrm{supp}(x) = S} \lVert y - A x \rVert_2
```

实现见 `cs/solvers.py::cosamp`。每轮做 4 次矩阵-向量运算。

> **⚠️ 本项目修复过的缺陷（见 `CHANGELOG.md` #2）**：top-k 必须按**系数幅值**取。
> 早期实现写成 `top = _support_of(xs)[:k]`，而 `_support_of` 返回的是
> **索引升序**数组，切片前 k 个与幅值无关 ⇒ NMSE约 `1.24`，**比猜零解还差**。
> 正确写法：
>
> ```python
> nz  = np.flatnonzero(np.abs(xs) > 1e-10)
> top = nz[np.argsort(-np.abs(xs[nz]))][:k]
> ```

### 复杂度

每轮 2 次 `Aᵀr` + 2 次 `A·x` + 2 次支撑集 LS，各`O(mn)`。
总**时间 `O(max_iter·mn)`**，**空间 `O(mn)`**。

### 已知失效条件

- 与 OMP 相同的相干性失效模式。
- `k ≥ n/2` 时无意义（代码中 `k` 被 clip 到 `n//2`）。
- 残差不降时提前停止：实现用 `‖r_new‖ ≥ ‖r‖ - 1e-12` 判据，可能在
  未改善时停在劣解上。

---

## 3. IHT — Iterative Hard Thresholding

> Blumensath & Davies (2009), *Iterative hard thresholding for sparse recovery*。

### 更新式

```math
x^{(j+1)} = \mathcal{H}_k\!\left( x^{(j)} + \frac{1}{\lVert A \rVert_2^2} A^\top (y - A x^{(j)}) \right)
```

其中 `‖A‖₂` 用**幂迭代**估计（`cs/operators.py::spectral_norm`，
**固定初始向量 `v = 1/√n·1` 以保证确定性**），`H_k` 是硬阈值算子：
保留幅值最大的 `k` 个坐标并**保留原值**（不做软收缩）。

步长 `1/‖A‖₂²` 保证目标函数 `½‖Ax−y‖²` 每轮单调不增。

### 超参

`k`（必需）、`max_iter`、`tol`、隐式 `‖A‖₂`（幂迭代 30 次，固定初始向量）。

**收敛条件**：`|prev_residual − cur_residual| < tol`，`tol` 默认 `1e-10`。
实测通常在 10~20轮内收敛。

### 复杂度

每轮 1 次 `A·x` + 1 次 `Aᵀr` + 1 次 `argpartition`（`O(n)`）。
总**时间 `O(max_iter·mn)`**，**空间 `O(mn)`**。

### 已知失效条件

- 硬阈值不做幅度收缩，配合支撑集 LS 去偏（`debias=True`）才能拿到好结果；
  不去偏则幅度系统性偏小。
- 对步长敏感：`spectral_norm` 低估会导致步长过大而发散（实现中用
  `max(‖A‖², 1e-12)`兜底）。
- **`m` 接近 `n` 时最小范数解会把 `y` 完全拟合** —— 这是 §5 中
  "过采样 LS 估σ 失败"的同一根因。

---

## 4. FISTA / LASSO

> Beck & Teboulle (2009), *A Fast Iterative Shrinkage-Thresholding Algorithm
> for Linear Inverse Problems*。

### 问题与更新式

求解

```math
\min_{x}\; F(x) = \tfrac{1}{2}\lVert A x - y \rVert_2^2 + \lambda \lVert x \rVert_1
```

Nesterov 加速迭代（`cs/solvers.py::fista`）：

```math
z^{(j)} = x^{(j)} + \frac{t^{(j)} - 1}{t^{(j+1)}}\left( x^{(j)} - x^{(j-1)} \right)
\quad
x^{(j+1)} = \mathcal{S}_{\lambda}\!\left( z^{(j)} - \tfrac{1}{\lVert A\rVert_2^2} A^\top (A z^{(j)} - y) \right)
```

其中 `t^{(j+1)} = ½(1 + √(1 + 4 t^{(j)²}))`，`S_λ` 是**软阈值**算子
`sign(v)·max(|v|−λ, 0)`。

**自适应重启**：`if F(x^{(j+1)}) > F(x^{(j)})` 则重置 `t=1`、`z=x`，
保证目标单调。

### λ 的取值 —— 本项目修复过的核心缺陷

```text
lam = lam_ratio · max|Aᵀ y|,     默认 lam_ratio = 0.02
```

> **⚠️ 本项目修复过的缺陷（`CHANGELOG.md` #3）**：
> 早期实现为 `lam = 0.05 · k · max|Aᵀy|`。**这个 `k` 是错的**：
> `max|Aᵀy|` 本身已随 `k` 单调增（k 越大相关峰越高），再乘 `k` 使 `λ`
> 远超信号幅度 ⇒ 软阈值把整向量压成全零。实测 **NMSE = 1.0、F1 = 0.0**，
> 等价于没有求解器。
>
> 配套：`BenchmarkConfig.lam_grid` 从 6 点扩到**13 点**
> （`0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3, 0.5, 0.8, 1.2`，
> 覆盖约 3 个数量级）。原 `0.02~0.8` 网格**整段落在全零解区**
> （`λ ≥ 0.3·max|Aᵀy|` ⇒ NMSE = 1.0），等于没调参。

### 复杂度

每轮 1 次 `A·z` + 1 次 `Aᵀr` + 1 次软阈值。
总**时间 `O(max_iter·mn)`**，**空间 `O(mn)`**。含去偏时额外 `O(mk + k³)`。

### 已知失效条件

- **相干性失效**：`‖Aᵀr‖` 内的伪相关使 ℓ1 球面切到错误位置。
- **`λ` 网格不覆盖有效区间**（见上）。
- 稀疏度极高（`k/n` 接近 1）时 ℓ1 严重失配，应改用 ℓ0 族（OMP/AMP）。

---

## 5. AMP — Approximate Message Passing

> Donoho, Maleki & Montanari (2009), *Iterative Thresholded Decoding for
> Approximate Message Passing*。

### 迭代形式（本项目采用的正确形式）

```text
r    = y - A·x + t          # (m,)   残差 + Onsager 项回投
x⁺   = soft(Aᵀ·r, λ)       # (n,)   软阈值（非线性）
t⁺   = A·(x⁺ - x)          # (m,)   Onsager 修正
```

数学形式：

```math
r^{(j)}   &= y - A x^{(j)} + t^{(j-1)} \in \mathbb{R}^m
\\x^{(j+1)} &= \eta_\lambda\!\left( A^\top r^{(j)} \right) \in \mathbb{R}^n
\\t^{(j)}   &= A\!\left( \eta_\lambda\!\left( A^\top r^{(j)} \right) - x^{(j)} \right) \in \mathbb{R}^m
```

其中 `η_λ` 为软阈值算子。**逐轮迭代**。

### ⚠️ Onsager 项的定义域：`t ∈ R^m`（本项目踩了 4 轮的坑）

**这是本项目调试成本最高的一处，必须写对。**

`t` 定义在**观测空间 `R^m`**，与残差 `r` 同维。理由：

1. **残差回投**：`r = y - A x + t` 中`t` 与 `y`、`A x` 同属观测空间，
   这样`r` 才有"残差"的物理含义（与真实观测在同一量纲）。
2. **维度必须匹配**：若 `t ∈ R^n`，则 `z = Aᵀr + t` 会触发
   `(256,)` vs `(96,)` 一类的广播错误 —— 本项目实际出现过 4 轮。

> **已否决的错误写法**：让 `t` 留在 `R^n`、写成 `z = Aᵀr + t`。
> 实测该写法在 `λ ≤ 0.3·max|Aᵀy|` 时**必然发散**（100% 抛
> `ConvergenceError`），在 `λ ≥ 0.6·max|Aᵀy|` 时恒返回全零解。
> **根因**：该写法让 `-Aᵀy` 项被重复计入。正确形式只保留
> `t = A·η(Aᵀr) - Aᵀy`，它恰好是 `η` 在 `Aᵀy` 处的线性化抵消。
>
> 另注：写成 `t_new = x_new - η(Aᵀy - AᵀA x)` **不成立**（维度和符号都错）。

### 阻尼与发散检测

```math
x^{(j+1)} \leftarrow \alpha\,\tilde{x}^{(j+1)} + (1-\alpha)\, x^{(j)},
\qquad \alpha = \mathrm{clip}(\texttt{damping}, 0, 1) \in [0,1]
```

`α = 1` 即标准 AMP；`α < 1` 引入惯性以抑制振荡（默认 `damping = 0.5`）。

**发散判据**：若 `x⁺` 含非有限值，或 `‖x⁺‖ > 50·‖y‖` ⇒ 抛
`ConvergenceError`（`cs/errors.py` E300 层）。

### ⚠️ 收敛判据：只看 `‖Δx‖`（本项目修复过的缺陷）

```text
delta = ‖x⁺ - x‖
若 delta ≤ tol · max(1, ‖x‖) ⇒ 收敛
```

> **⚠️ 本项目修复过的缺陷（`CHANGELOG.md` #5）**：
> 早期判据把 Onsager 项的变化也算进去（`‖Δx‖ + ‖Δt‖`）。但 `t` **每轮都在变**，
> 该量恒不衰减 ⇒ 实测**第 2 轮即误判收敛**（或反向恒不收敛）。
> 修复：判据**只看 `‖Δx‖`**。

### 复杂度

每轮 1 次 `A·x` + 1 次 `Aᵀ·r` + 1 次 `A·(x⁺−x)` + 软阈值。
总 **时间 `O(max_iter·mn)`**，**空间 `O(mn)`**（比 FISTA 每轮多一次 `A·`）。

### 已知失效条件

- **iid 高斯假设不成立时状态扩展失效**：本`fourier` / `coherent` regime
  下 `A` 高度结构化，AMP 的标量状态扩展理论**不适用**，实测接近或等于猜零解。
- **`λ` 过小 ⇒ 发散**（见上述错误写法的实测）。
- **`damping` 过小（→0）⇒ 振荡**；过大（→1）⇒ 收敛极慢。
- 去掉 Onsager 项（`t ≡ 0`）后 AMP **退化为 IHT**，此时失去理论保证。

---

## 6. 重加权家族（IRL1 / Candès–Wakin–Boyd）

> Candès, Wakin & Boyd (2008), *Enhancing Sparsity by Reweighted ℓ1
> Minimization*, J. Fourier Anal. Appl. —— 迭代重加权

```math
w_i^{(j)} = \frac{1}{|x_i^{(j)}| + \varepsilon}
\quad\Longrightarrow\quad
\sum_i w_i^{(j)}\,|x_i|
```

重复求解加权 ℓ1 会**二次方地**逼近 ℓ0 约束，在相变区严格优于单次 ℓ1。
本项目三个实现：

| 求解器 | 外层 | 内层 |
| --- | --- | --- |
| `reweighted_fista` | 重加权（`outer=4`） | FISTA（L1 外层 × FISTA 内层） |
| `irl1_omp` | 支撑投票 `w_i = Σ_j 1{第 j 轮选中 i}` | OMP 加权相关度 `\|a_iᵀr\| · w_i` |
| `amp_rl1` | 重加权（`outer=3`） | AMP（含 Onsager） |

### 超参

`outer`（外层轮数）、`eps_rel`（稳定化常数，取 `eps_rel·max|x|`，
过小会数值爆炸）、`lam_scale`（正则强度整体缩放）。

### 复杂度

- `reweighted_fista`：**`O(outer · max_iter · mn)`**
- `irl1_omp`：每轮 1 次 `A·x` + 1 次 `Aᵀr` + 1 次 argpartition ⇒ **`O(outer · mn)`**
- `amp_rl1`：**`O(outer · max_iter · mn)`**

三者空间均为 **`O(mn)`**。

### 已知失效条件 —— 重要负面结论

- **`reweighted_fista` 在 `m < n` 欠定设定下劣于 ℓ1 基线。**
  实测（6 regime 逐实例求和）：`rw_fista` SUM = `7.455e-01`
  vs `iht` SUM = `2.311e-01` —— **劣化 3.2 倍**。
  根因：加权后软阈值的有效支撑集在欠定下严重膨胀。实测单实例
  （`wellcond`, m=96, n=256, k=12）`rw_fista` 的`n_supp = 139`
  （`k_true = 12`），NMSE `3.93e-01`，且**对 `k` 完全不敏感**
  （`k = 4 / 8 / 12 / 16 / 24 / 30` 全部得到同一结果 `3.9310e-01`）——
  这正是它无法在本设定下工作的直接证据。
- `amp_rl1` 实测**在所有 6 个 regime 上 NMSE恒为 `1.000`、F1 = `0.000`**，
  即完全退化为猜零解。
- 重加权理论保证是**渐近**的，在有限样本 + 欠定 + 未标定 `ε` 时不成立。

---

## 7. k 估计：不一致性原理

### Morozov 不一致性原理

```math
k_{\mathrm{hat}} = \arg\min_{k \in \mathcal{K}} \left| \lVert r_k \rVert_2^2 - \tau\, m\, \hat{\sigma}^2 \right|,
\qquad \mathcal{K} = (4, 6, 8, 10, 12, 14, 16, 20, 24, 30)
```

- `k` 偏小 ⇒ 欠拟合 ⇒ 残差**远高于**噪声水平
- `k` 偏大 ⇒ 把噪声也拟合了 ⇒ 残差**低于**噪声水平
- 真值落在"噪声拐点"处

`τ` 为残差目标倍数：`τ < 1` 偏保守（欠拟合），`τ > 1` 偏激进。
校准值 **`τ = 0.55`** 可把实测 `k_hat` 拉回 `k_true` 附近。

### σ 估计：四次尝试，只有第四次可用

`cs/kselect.py::estimate_sigma2` 的完整实测记录（详见
`docs/architecture.md` §4.6）：

| # | 方案 | σ̂/σ_true 实测 | 结论 |
| --- | --- | --- | --- |
| 1 | OMP 残差 MAD | `0.075 ~ 0.163`，6/6 全低估 | ❌ OMP 把噪声拟合进支撑集，残差里是信号拟合误差 ⇒ `k_hat` 恒为 30 |
| 2 | 过采样 LS（m×(m+p)） | ≈ `0`，残差塌到 `3.5e-15` | ❌ Gram 数值秩亏，最小范数解完全拟合 `y` |
| 3 | 全局最小范数留出块 | `184 ~ 262`（高估 200 倍） | ❌ 欠定下全局 LS 连信号本身都外插错；6 折不降 ⇒ 维数亏，非折数问题 |
| 4 | **支撑集内插留出残差**（采用） | `1.26 ~ 1.70`（高估 26%~70%） | ⚠️ 仍不精确，但偏差方向**安全** |

采用方案（`n_folds=4`）：训练折上跑 OMP(k₀) 得支撑 `S`，**只在 `S` 的 k₀ 个
坐标上**做 LS，用这些系数内插留出块。因为 `k₀ ≪ m_folds`，训练折内插**超定**，
无外插误差。

### 复杂度

`σ` 估计：`n_folds` 次 OMP ⇒ **`O(n_folds · k₀ · mn)`**。
`k` 估计：`|K| = 10` 个候选各跑一次 OMP ⇒ **`O(|K| · k · mn)`**。
**总成本约等于跑 40+ 次 OMP** —— 这是 `cs_fuse` 单实例墙钟远高于
单个 OMP 的主要原因。

### 已知失效条件

- **σ̂ 仍高估 26%~70%** ⇒ `k_hat = 4~10` 系统性**偏保守**
  （`k_true = 10~12`）。偏差方向安全，但不得声称"k 估计精确"。
- `low_snr` / `heavy_tail` regime 下偏差最明显（实测 `low_snr` 得 `k_hat=4`
  vs `k_true=10`）。
- 候选网格固定为 `(4, 6, ..., 30)`：**若真值 k 超出该区间（`k > 30` 或 `k < 4`），
  必然越界**。这是已知的结构性限制。

---

## 8. 支撑集后处理

### 去偏 `debias_refit`

```math
\hat{x}_S = \arg\min_{x_S} \lVert y - A_S x_S \rVert_2, \qquad \hat{x}_{S^c} = 0
```

贪心/阈值类方法在支撑集上通常只保证"稀疏解"，重拟合去掉阈值带来的幅度收缩偏差，
是 NMSE 改进的主要来源之一。

**可验证不变量**：同一支撑集上 LS 是最小二乘最优 ⇒
**`‖y − A x̂‖₂ ≤ ‖y − A x_prev‖₂`**。

### 1-opt 精修 `polish_support`

每轮交换"最弱支撑点"与"最大残差相关非支撑点"，**仅在降低训练残差时接受**。

> **⚠️ 实测过拟合（`CHANGELOG.md` #8）**：`m < n` 时训练残差是**有偏的模型
> 选择信号**，该步骤实测**比最优单法差 +229%**。因此**默认关闭**
> （`use_polish=False`），仅保留为消融项 A4。开启时限制 1 轮且要求相对
> 下降 > 5%。

**可验证不变量**：`‖y − A_S x_S‖²` **单调不增**（代码里每次接受都验证
`val < best * 0.95`）。

---

## 9. 复杂度汇总

| 求解器 | 时间复杂度 | 空间|每轮矩阵-向量运算数 |
| --- | --- | --- | --- |
| OMP | `O(k·mn)` | `O(mn)` | 2 |
| CoSaMP | `O(max_iter·mn)` | `O(mn)` | 4 |
| IHT | `O(max_iter·mn)` | `O(mn)` | 2 |
| FISTA | `O(max_iter·mn)` | `O(mn)` | 2 |
| AMP | `O(max_iter·mn)` | `O(mn)` | 3 |
| `reweighted_fista` | `O(outer·max_iter·mn)` | `O(mn)` | 2 × outer |
| `irl1_omp` | `O(outer·mn)` | `O(mn)` | 2 |
| `amp_rl1` | `O(outer·max_iter·mn)` | `O(mn)` | 3 × outer |
| σ 估计 | `O(n_folds·k₀·mn)` | `O(mn)` | — |
| k 估计 | `O(|K|·k·mn)` | `O(mn)` | — |
| `cs_fuse` 合计 | `O((|K| + outer·max_iter)·mn)` | `O(mn)` | — |

**统一瓶颈**：所有求解器的瓶颈都是 `A·x` 与 `Aᵀ·r` 两个矩阵-向量乘法，
均为 `O(mn)`。在 `m=96, n=256` 下 `mn = 24576`，纯 numpy BLAS 实测单次
`O(10⁵)` flops 级别，因此**迭代次数**而非单次运算量决定墙钟时间。

**实测墙钟**：18 实例 × 12 方法全跑 `wall_sec = 4.56`
（`cli.py bench --seeds 3`，`m=96, n=256`）。

---

## 10. 已修复缺陷汇总索引

完整的 8 项缺陷记录在 `CHANGELOG.md`。在本文档中的位置：

| # | 缺陷 | 本文档章节 |
| --- | --- | --- |
| 1 | Fourier 矩阵严格重复列 | `docs/architecture.md` §4.4 |
| 2 | CoSaMP top-k 取错 | §2 |
| 3 | FISTA/AMP 全零解（λ 量纲） | §4 |
| 4 | AMP Onsager 定义域（4 轮） | §5 |
| 5 | AMP 收敛判据误触发 | §5 |
| 6 | OMP 残差 MAD 低估 σ | §7 |
| 7 | 过采样 LS / 全局最小范数留出块失败 | §7 |
| 8 | 训练残差局部搜索过拟合 | §8 |