# tests — SparseForge 测试套件

压缩感知 / 稀疏恢复系统的 pytest 测试套件。**75 条可验证不变量** + 契约/边界/兜底用例。

作者：晨星 · CJX0712

---

## 运行方式

```bash
cd C:/Users/Administrator/WorkBuddy/2026-10-05-03-16-42/sparseforge

# 全量
"C:/Users/Administrator/.workbuddy/binaries/python/envs/sparseforge/Scripts/python.exe" -m pytest tests/ -q

# 跳过子进程慢测（CLI 冒烟里check --seeds 3 约 15s）
... -m "not slow"

# 覆盖率
... -m pytest tests/ -q --cov=core --cov=cs --cov=eval --cov=data --cov-report=term-missing

# 风格门禁
... -m ruff check tests/
... -m ruff format --check tests/
```

## 实测结果（本机真实数字）

| 项 | 结果 |
|---|---|
| pytest | **165 passed, 4 xfailed, 0 failed**（47.99s） |
| 不变量测试数 | **75**条 `test_invariant_*` |
| `ruff check tests/` | **All checks passed!** |
| `ruff format --check tests/` | **11 files already formatted** |

### 覆盖率（`core` / `cs` / `eval` / `data`）

| 模块 | Stmts | Miss | Cover |
|---|---:|---:|---:|
| `core/config.py` | 94 | 3 | **97%** |
| `core/errors.py` | 18 | 0 | **100%** |
| `core/interfaces.py` | 14 | 14 | 0%（见下） |
| `core/seed.py` | 30 | 0 | **100%** |
| `core/types.py` | 71 | 0 | **100%** |
| `cs/fusion.py` | 151 | 6 | **96%** |
| `cs/kselect.py` | 67 | 2 | **97%** |
| `cs/operators.py` | 48 | 1 | **98%** |
| `cs/reweighted.py` | 109 | 12 | **89%** |
| `cs/solvers.py` | 211 | 7 | **97%** |
| `data/generators.py` | 85 | 0 | **100%** |
| `eval/metrics.py` | 60 | 0 | **100%** |
| **TOTAL** | **958** | **45** | **95%** |

目标 ≥80%，实测 **95%**。`core/interfaces.py` 的 0% 不计入concern：它是四个
`Protocol` 的纯类型声明（全部是 `def f(...) -> ...: ...`），运行时无可执行语句，
覆盖率工具无法计入。

---

## 文件结构

| 文件 | 不变量数 | 覆盖内容 |
|---|---:|---|
| `conftest.py` | — | `solver_fn`（参数化）、`tiny_instance`、`rng`、`noiseless_problem`、autouse 熵隔离、`slow` 标记注册 |
| `test_operators.py` | 10 | 正交性、软阈值非扩张、尺度不变性、幂迭代一致性、去偏单调性、`oracle_nmse` |
| `test_solvers.py` | 8 | 精确恢复、解向量合法性、debias 单调性、防泄漏签名、AMP 发散（xfail） |
| `test_kselect.py` | 7 | 噪声估计无偏性、k 网格合法性、频率归一化（xfail） |
| `test_metrics.py` | 11 | `beats` 定义符合性、`aggregate` 守恒、F1/NMSE 语义、阈值判据 |
| `test_determinism.py` | 11 | 逐位可复现、`_kind_offset` 稳定性、熵入口一致性、pipeline 两次一致 |
| `test_errors.py` | 7 | 错误码E100–E500 稳定、上下文格式化、19 项配置校验拒绝 |
| `test_generators.py` | 9 | SNR 自洽性、列归一化、块结构、傅里叶无重复列、spike 映射（xfail） |
| `test_fusion.py` | 12 | `polish_support` 单调性、路由划分、消融开关、旗舰契约 |
| `test_cli.py` | — | 5 个子命令 subprocess 冒烟（含 3 项 CLI 缺陷锁定） |

---

## 已知不成立的不变量（**未写成测试**，原因如下）

任务书列出的部分不变量在**当前实现下数学上不成立**。按"禁止写假绿测试"的要求，
这些**没有**写成断言，而是在此记录原因。每条都在对应文件里保留了**可验证的替代命题**。

### 1. `soft_threshold`幂等性 ❌

**断言**：`‖soft(soft(v,λ),λ) − soft(v,λ)‖∞ < 1e-15`
**实测**：误差 = **1.3**（λ=1.3, v 幅度 ~5）

软阈值**不是**幂等算子。对任意 `|v| > λ`，`|soft(v,λ)| = |v| − λ > 0`，二次收缩
再减一次 λ。精确恒等式是 `soft(soft(v,λ),λ) == soft(v,2λ)`（已写成断言）。
真正幂等的是**硬**阈值。docstring 只声称 `|S(v)| ≤ |v|`（非扩张，成立）。

→ 替代测试：`test_invariant_soft_threshold_is_not_idempotent`、
`test_invariant_soft_threshold_is_nonexpansive`

### 2. 40dB 高 SNR 下 `OMP nmse < 1e-10` ❌

**断言**：m=64, n=256, k=4, SNR=40dB 上 `nmse < 1e-10` 且 `support_f1 == 1.0`
**实测**：nmse = **1.64e-5**，f1 = 1.0

带噪时 NMSE 有信息论下界＝噪声能量比。40dB ⇒ 下界 ≈ 1e-4，**数学上不可能**
达到 1e-10。实测 1.64e-5 与该下界一致——误差来自**噪声**而非算法。
`support_f1 == 1.0` 单独成立。

→ 替代测试：改用**无噪声**实例（`nmse < 1e-10`，20 seed 最差 3.5e-29），
外加 SNR 线性下降性质（40→60dB 使NMSE 降约 100 倍）
→ `test_invariant_omp_exact_recovery_noiseless`、
`test_invariant_omp_exact_recovery_robust_over_seeds`、
`test_invariant_omp_high_snr_hits_noise_floor`

### 3. `estimate_sigma` 无偏性（默认口径） ❌

**断言**：纯噪声下相对误差 < 30%
**实测**：默认 `k0` 下最差 **85%**，60/60 例**全部高估**（ratio 1.0–1.85）

`estimate_sigma2` 的内层 `for k in k_grid` 循环也执行 `freq[sup] += 1`，
累加 `n_boot × |k_grid|` 次却只除以 `n_boot`。根因是支撑维度接近折内训练
样本数时内插退化。**注意**：这与 `cs.fusion.stability_select`（无 k 内层循环）
形成对照——后者的 freq 严格落在 [0,1]。

→ 替代测试：`k0=1` 时最差 20.9% < 30%（成立，写为
`test_invariant_estimate_sigma_is_unbiased_on_pure_noise`）；
默认口径的高估偏差**如实锁定**为一条独立测试
→ `test_known_bias_estimate_sigma_default_k0_overestimates`

### 4. `stability_k` 频率归一化 ❌ → xfail

**断言**：`freq ∈ [0,1]`
**实测**：`freq.max()` = **10.0**（m=64）、7.75（m=48）

**缺陷 D1**（详见下表）。写成 `xfail(strict=True)`。

### 5. `spike` 增大 ⇒ 相干度升高 ❌ → xfail

**断言**：`coherence(A(spike=0.99)) > coherence(A(spike=0.90))`
**实测**：两者**完全相同**（0.5136 / 0.5454）

**缺陷 D3**（详见下表）。写成 `xfail(strict=True)`。

### 6. `polish_support` 保持正确支撑集不变 ❌

**断言**：正确支撑集精修后残差变化 < 5%
**实测**：支撑集被改动 `[37,59,70,72,89]→[37,59,70,89,91]`，
残差 3.190e-3 → 3.022e-3（降 5.27%，**刚好越过** 5% 接受门槛）

这正是 docstring 诚实披露的"``m < n`` 时会**过拟合**"。5% 相对门槛在
`m ≪ n` 时不足以拦住噪声驱动的伪改善。

→ 替代测试：改测**真正成立**的结构性质（1-opt 交换 ⇒ 支撑集大小恒定、
与原集至少共享 size−1 个）+ 单调性；过拟合行为作为**已知缺陷**锁定
→ `test_polish_support_preserves_support_size`、
`test_known_defect_polish_support_overfits_on_correct_support`

### 7. `spectral_norm` 30 步即精确 ❌（默认参数）

**断言**：相对误差 < 1e-5
**实测**：从 6e-16 到 **5.5e-2** 不等

默认 `n_iter=30` 的精度**强烈依赖谱间隙**。最差样本 `(10,25)` 间隙仅
`(σ₁−σ₂)/σ₁ ≈ 0.04`，30 步远未收敛（`n_iter=300` 即精确到 1e-16）。

→ 替代测试：对**有谱间隙**的矩阵族（正交列/秩一/对角）断言 30 步即机器精度；
`n_iter=300` 时所有形状相对误差 < 1e-7
→ `test_invariant_spectral_norm_matches_dense_svd`

### 8. `nmse` 参数对称 ❌

**断言**：`nmse(a,b) == nmse(b,a)`
**实测**：2.053 vs 2.910

NMSE 分母是 `‖x_true‖²`（第二个参数），**按设计不对称**。

→ 替代测试：断言"分母跟随真值"的正确性质 + 缩放 x_true 时分子分母同变
（**不存在** `1/c²` 律）→ `test_invariant_nmse_denominator_is_x_true_norm`

### 9. 退化输入必抛 `NumericsError`/`ConfigError` ❌

**断言**：A 全零 / y 全零 / n=1 必抛异常
**实测**：**全部不抛**，正常返回空支撑解

OMP 的相关性 `max|Aᵀr|` 为 0 时，循环在 `corr[i] <= 0` 处**正常退出**——
这是**优雅降级**而非异常。`m < k`（m=4, k=6）同理不崩。

→ 替代测试：断言"不抛裸异常 + 输出合法（形状/有限性）"；
真正的异常路径另行覆盖（`k=None` → NumericsError、支撑集越界、
`check_finite` 的 NaN/Inf、`make_instance` 的 5 项形状校验）
→ `test_degenerate_inputs_do_not_crash`、`test_m_less_than_k_is_handled_without_crash`

---

## 发现并 xfail 标注的实现缺陷

| # | 位置 | 问题 | 影响 | 测试 |
|---|---|---|---|---|
| **D1** | `cs/kselect.py:164-180` | `stability_k` 的 `freq` 未归一化：内层 `k_grid` 循环也执行 `freq[sup] += 1`，累加 `n_boot×\|k_grid\|` 次却只除 `n_boot` | freq 被放大 10 倍（实测 max=10.0）；`stable` 几乎命中所有坐标 ⇒ `k_best` 退化为"稳定坐标总数"，被 clip 到 `m−2`（m=48 时恒为 46，完全饱和）⇒ **`stability_k` 实际失去 k 估计能力** | `test_invariant_stability_k_freq_is_normalized` (xfail) + `test_known_defect_stability_k_saturates_at_m_minus_2` |
| **D2** | `cs/solvers.py:252`、`cs/reweighted.py:174` | `amp` 在良态 wellcond 实例上 **100% 发散**（m∈{24..96} × 6 seed = 30/30 全 `ConvergenceError`）；`amp_rl1` 同样发散但抛**裸 `RuntimeError`**，逃出 `SparseForgeError` 体系 | AMP 家族实际不可用（被 pipeline 的 `except Exception` 静默吞掉）；裸异常破坏调用方的按语义捕获 | `test_xfail_amp_diverges_on_wellcond` (xfail) + `test_xfail_amp_rl1_raises_raw_runtime_error` (xfail) + `test_amp_rl1_violates_error_hierarchy` |
| **D3** | `data/generators.py:58-74` | `make_spiked_matrix` 的 `spike→相干度`映射**与docstring 声明相反**。`v=(1−s)a_i+s a_j` 在 `a_i⊥a_j` 时给出 `cos=(1−s)/√((1−s)²+s²)`，随 `s` **单调下降**（s=0.1→0.994, s=0.9→0.110）。要让两列共线应用**小** `s` | `make_instance` 只传 `s∈[0.90,0.98]`（曲线极平坦的膝点），使 `coherence∈{0,0.5,1.0}` 产出**逐位相同**的矩阵 ⇒ **`coherence` 参数完全失效**，`coherent` regime 未真正制造相干硬实例 | `test_invariant_spike_parameter_increases_coherence` (xfail) + `test_known_defect_coherence_param_has_no_effect` |
| **D4** | `cli.py:119` | `check` 子命令 `--seeds` 默认值 **2**，但 `SparseForgePipeline.__init__` 要求 `n_seeds >= 3`（抛 E200） | `python cli.py check`（**不带参数**）永远以裸 traceback 收场 | `test_cli_check_default_seeds_is_broken` + `test_cli_check_rejects_below_three_seeds[1,2]` |
| **D5** | `cli.py:84,92,102` | `ablate`/`failures`/`demo` 依赖 `examples.ablate`/`examples.failures`/`examples.run_demo`，而 `examples/` 目录下**只有 `README.md`** | 三个子命令全部 `ModuleNotFoundError` 失败（rc=1），CLI 实际只有 `bench`/`check` 可用 | `test_cli_subcommands_failing_on_missing_examples[ablate/failures/demo]` |
| **D6** | `preprocess/transforms.py:49` | `gram_whitener` 用 `np.eye(m)` 构造收缩项，但 `G = AᵀA` 是 `(n,n)` | `m≠n` 时**必然** `ValueError: broadcast shape mismatch`；`TransformPipeline` 默认（center+whiten）在压缩感知场景（`m<n`）**完全不可用** | 未写入`tests/`（`preprocess` 不在本次覆盖率目标内，见下方说明） |

> **关于 D6**：`preprocess` 不属于本次要求的覆盖率目标包（`core`/`cs`/`eval`/`data`），
> 故未为其建测试文件。但该缺陷会使 `TransformPipeline()` 在任何 `m<n` 的实例上
> 崩溃，属P0 级问题，建议单独修复：`np.eye(m)` 应为 `np.eye(n)`。
> 顺带发现：`shrink=0` 时 `W = inv(chol(AᵀA))` 也**不**满足 docstring 声称的
> `(AW)ᵀ(AW) ≈ I`（实测 m=n=6 时偏差 74.5）——白化方向写反了，
> 应为 `W = inv(chol(G))ᵀ`。

---

## 确定性保证

- **autouse fixture** 每个测试前调`set_all(DEFAULT_SEED)`，隔离全局熵状态
- 实例可复现性用 `np.array_equal`（**逐位**）而非 `allclose`
- `_kind_offset(kind, seed)` 稳定性专门测试：6 个 kind 两两不撞车、跨调用一致、
  offset 非负且随 kind 单调递增（`hash()`因 PYTHONHASHSEED 随机化被刻意弃用）
- `pipeline.run(methods=["omp"])` 两个独立 Pipeline 实例跑出的 NMSE 逐行相同

## 已知行为（非缺陷，但与直觉不符）

- **`clustered` 在 `k < block=8` 时激活 1 块**（8 个非零），因为
  `round(k/8)=0` 被 `max(1, ...)` 抬高。调用方不应假设"支撑大小 == k"。
  → `test_known_quirk_clustered_ignores_small_k`
- **FISTA 的 `meta["objective"]` 是 debias 之前的值**，与去偏后 `x_hat` 的目标值
  不同（fista 内部硬编码 `debias=True`，忽略 `cfg.debias`）。验证需重算而非比对。
- **`nmse` 按设计不对称**（分母是 `‖x_true‖²`）。
- **`SNR` 有随机波动**：噪声实际范数是 `sigma·‖randn(m)‖`，非确定值。偏差
  ~1/√(2m)。故 SNR 自洽性测试用 `m=400` 并断言 < 1dB（实测最差 0.50dB），
  而非断言精确相等。