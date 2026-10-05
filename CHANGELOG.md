# Changelog

本文件遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 规范，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

## [0.1.0] - 2026-10-05

首个可复现版本。核心算法链路（数据生成 → 11 个求解器 → 3 类指标 → 门禁判定）
已打通，**旗舰 `cs_fuse` 未通过性能门禁，质量等级评定为 C**。

### 修复的算法缺陷

以下 8 项均为本项目实测定位并修复的**真实算法缺陷**，不是代码风格问题。
每项都附修复前后的实测数字。详见 `docs/algorithm_spec.md` 与
`cs/fusion.py`、`cs/kselect.py`、`cs/solvers.py` 的模块 docstring。

#### 1. Fourier 测量矩阵产生严格重复列（结构性算子失效）

- **根因**：`Re(exp(iθ))` 中 `cos` 是偶函数，频率 `+r` 与 `-r` 的实部**完全相同**，
  因此矩阵含严格重复列。
- **实测**：相干度 `1.0000`、条件数 `6.47e13`，OMP / IHT / FISTA 全部崩溃，
  aggregate NMSE `1.469`（比返回零解还差 47%）。
- **修复**（`data/generators.py::make_fourier_matrix`）：改为堆叠 `cos` + `sin`
  的标准实值化 `A = [cos(Θ); sin(Θ)] / sqrt(2·half)`，既保留部分傅里叶的 RIP
  结构，又消除重复列。
- **修复后**：该 regime NMSE `2.30e-3`、F1 `0.92`。

#### 2. CoSaMP top-k 取错，导致输出比猜零解还差

- **根因**：`top = _support_of(xs)[:k]` 取的是**支撑索引升序的前 k 个**，
  而非**系数幅值最大的 k 个**。`_support_of` 已按索引升序返回，切片前k 个
  完全与系数大小无关。
- **实测**：NMSE约 `1.24`，劣于直接返回零解（NMSE = 1.0）。
- **修复**（`cs/solvers.py::cosamp`）：改为
  `nz = flatnonzero(|xs| > 1e-10)` → `top = nz[argsort(-|xs[nz]|)][:k]`。

#### 3. FISTA / AMP 全部退化为全零解（正则强度量纲错误）

- **根因**：原式 `lam = 0.05 · k · max|Aᵀy|`。`max|Aᵀy|` **本身已随 k 单调增**
  （k 越大相关峰越高），再乘一次k 就使 `lam` 远超信号幅度，软阈值把整向量压成零。
- **实测**：NMSE = `1.000`、F1 = `0.000`，即完全无效。
- **修复**（`cs/solvers.py::_default_lam`）：改为 `lam = 0.02 · max|Aᵀy|`
  —— **不再乘 k**，并在 docstring 中写明原因，防止回归。
- **配套修复**：`BenchmarkConfig.lam_grid` 从 6 点扩到 **13 点**
  （0.002 → 1.2，覆盖约 3 个数量级）。原 0.02~0.8 网格**整段落在全零解区**
  （`lam ≥ 0.3·max|Aᵀy|` ⇒ NMSE = 1.0），等于没调参。

#### 4. AMP Onsager 修正项的定义域反复搞反（历经 4 轮）

- **根因**：Onsager 项 `t` 的定义域在 `R^n` 与 `R^m` 之间反复摇摆，产生
  `(256,)` 与 `(96,)` 的广播错误。
- **最终确定的正确形式**：`t ∈ R^m`（观测空间）

  ```text
  r     = y - A·x + t            # (m,)   残差 + Onsager 回投
  x⁺    = soft(Aᵀr, λ)           # (n,)   软阈值
  t⁺    = A·(x⁺ - x)            # (m,)   Onsager 项
  ```

- **已否决的错误写法**：让 `t` 留在 `R^n`、写成 `z = Aᵀr + t`。实测
  `lam ≤ 0.3·max|Aᵀy|` 时**必然发散**（100% `ConvergenceError`），
  `lam ≥ 0.6·max` 时恒返回全零解。根因是 `-Aᵀy` 被重复计入。

#### 5. AMP 收敛判据误触发（Onsager 项污染 delta）

- **根因**：收敛判据里混入了每轮都在变的 Onsager 项，导致 `delta` 恒不衰减
  —— 实测**第 2 轮即误判收敛**。
- **修复**（`cs/solvers.py::amp`）：收敛判据**只看 `‖Δx‖`**，
  Onsager 项的变化不参与判定。

#### 6. OMP 残差 MAD 低估噪声 σ 一个数量级

- **根因**：OMP 把噪声一起拟合进了支撑集，因此残差里是**信号拟合误差**而非噪声。
- **实测**：`σ̂/σ_true` ratio 仅 `0.075 ~ 0.163`，**6/6 regime 全部低估**。
  后果：σ̂ 过小 ⇒ 不一致性原理的残差目标被压低 ⇒ `k_hat` **恒为网格最大值 k=30**。
- **修复**（`cs/kselect.py::estimate_sigma2`）：改用**支撑集内插留出残差**
  （先在训练折跑 OMP 得支撑 S，再只在 k0 个坐标上做 LS 内插留出块）。
  实测 σ̂ 仍有 `26% ~ 70%` 高估，但偏差方向**由系统性低估翻转为安全侧高估**，
  使 `k_hat` 落回4~10 的可用区间。

#### 7. 过采样 LS 与全局最小范数留出块估σ 均告失败

两种备选方案被实测否决，过程记录在 `cs/kselect.py::estimate_sigma2` docstring：

| 方案 | 实测 ratio（σ̂/σ_true） | 失败原因 |
| --- | --- | --- |
| 过采样 LS（m×(m+p)） | ≈ `0`（残差塌到 `3.5e-15`） | `m` 与 `m+p` 接近时 Gram **数值秩亏**，最小范数解把 `y` 完全拟合 |
| 全局最小范数留出块 | `184 ~ 262`（**高估约 200 倍**） | 纯噪声上无偏（ratio 0.79~1.02 ✅），但 `m<n` 欠定时全局 LS 连**信号本身**都外插错；6 折几乎不降 ⇒ 不是折数问题，是**维数亏** |

#### 8. 训练残差局部搜索在m<n 时过拟合

- **根因**：`m < n` 时训练残差是**有偏的模型选择信号** —— 自由度越多，对训练
  数据的解释力越强，局部搜索必然朝过拟合方向走。
- **实测**：比最优单法差 **+229%**。
- **修复**：`polish_support`（1-opt 精修）**默认关闭**（`use_polish=False`），
  仅保留为消融项 A4；开启时限制为1 轮，且要求相对下降 > 5% 才接受交换。
  `hpo.search.grid_search_ratio` 相应改用**不一致性原理**作为评分口径。

### 新增

- 6 种DGP regime 的准合成数据生成器（`wellcond` / `coherent` / `low_snr` /
  `clustered` / `fourier` / `heavy_tail`），覆盖易到难，避免单一 regime 独大。
- 11 个求解器：Tier-1 纯 numpy 5 个（OMP / CoSaMP / IHT / FISTA / AMP）、
  重加权家族 3 个（`reweighted_fista` / `irl1_omp` / `amp_rl1`）、
  Tier-0 sklearn 3 个（`sk_omp` / `sk_lasso` / `sk_lars`）。
- 真值无关的稀疏度估计（`cs/kselect.py`）：不一致性原理 + 支撑集内插 σ 估计 +
  稳定性选择。
- 预注册门禁阈值（`core/types.py::GATE_THRESHOLDS`），方案阶段定死、
  **禁止事后调低**。
- 统一熵入口 `core/seed.py::set_all()`，禁止任何模块另起`RandomState`。
- 异常层级 `core/errors.py`：E100 数据生成 / E200 数值 / E300 收敛 /
  E400 配置 / E500 后端。
- CLI 五个子命令：`bench` / `check` / `demo` / `ablate` / `failures`。
- 确定性自检：`cli.py check` 同 seed 跑两次逐位比对（实测 216 行 bit_identical=True）。

### 已知限制

- **旗舰 `cs_fuse` 未通过性能门禁**：aggregate NMSE `9.524e-01`，劣于最强基线
  `omp` 的 `7.057e-02`（劣化 13.5倍）。GATE1 / GATE2 / GATE4 / GATE5 全部 FAIL。
  已定位根因为**路由阈值未按维数标定**（详见 `docs/model_card.md`），正在修复。
- `rw_fista` 在 `m<n` 欠定设定下劣于 ℓ1 基线（实测 SUM `7.455e-01`
  vs IHT `2.311e-01`）。
- `amp` / `amp_rl1` 在本设定下接近或等于猜零解（`amp_rl1` NMSE 恒为 `1.000`）。
- 所有 regime 下 `success_rate = 0`：这是**门禁口径问题**（要求支撑集 F1
  完全恢复 = 1.0，噪声设定下未达到），非方法失效。详见 `docs/model_card.md`。
- 测试套件尚未落地（`tests/` 只有 README），`pytest` 当前收集 0 个用例。

[Unreleased]: https://github.com/CJX0712/sparseforge/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/CJX0712/sparseforge/releases/tag/v0.1.0