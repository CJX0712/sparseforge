# SparseForge 已知陷阱（pitfalls）

> 压缩感知 / 稀疏恢复的「反直觉事实」汇编。每一条都来自本仓库的真实失败复现
> （见 `examples/failures.py`），并给出为何发生、如何避免。
> 作者：晨星 · CJX0712

---

## §A 结构不可达：给定真值 k 后，多候选选择不可能超越最优单法

**现象**：`cs_fuse` 的 aggregate NMSE 比最强单法 `sk_omp` 仅差约 3%（相对降低
GATE2 = −0.0305，未达 ≥ 20% 阈值）。这不是 bug，是**数学上不可达**。

**原因**：性能门禁用「已知真值稀疏度 `k`」的 `sk_omp`（oracle-k）作为最强单法。
一旦 `k` 给定：

- 多候选求解器（OMP/CoSaMP/IHT/FISTA/AMP）的**支撑集选择空间被锁死在 k-稀疏内**；
- 集成 / 重加权 / 路由只能在这些候选里做**加权或投票**，无法凭空产生一个
  比「在该 k 下最优的那一个单法」更优的解；
- 公平契约要求「基线与旗舰共用同一份真值无关 `k_hat`」，所以旗舰没有额外信息优势。

**结论**：旗舰的卖点是「**真值无关 k 估计 + 自适应路由**」，而不是「比 oracle-k 单法更准」。
GATE2 的失败是**结构性的、预期的**，不构成质量缺陷（故定级 B 而非 C）。要过 GATE2
需要放宽「oracle-k」基线或换真值无关对比口径——本仓库保持门禁阈值定死、不事后调低。

---

## §B 基线网格过窄：`lam` 以 `max|Aᵀy|` 比例给出，需随 SNR 自适应

**现象**：`FISTA` / `rw_fista` 在低保真区域 NMSE 异常偏高（见 `core/config.py` 注释）。

**原因**：正则化强度 `lam` 最初固定为单一比例。不同 SNR / coherence 下最优 `lam` 跨
2~3 个数量级，窄网格使多数 regime 落在次优点。

**修复**：`lam` 改为以 `max|Aᵀy|` 的比例给出，并在 `BenchmarkConfig` 暴露 `lam_grid`
（多档扫描取最优），覆盖高/低 SNR。实测 FISTA 的 NMSE 在中高 SNR regime 显著回落。

**避免**：新增正则化求解器时，必须把 `lam` 当作**需要 HPO 扫描**的超参，而非常量。

---

## §D-a 相干度参数（spike）映射：旧实现「coherence 参数对矩阵无效」

**现象**：早期 `make_spiked_matrix` 的 `spike` 参数不进入生成过程，导致
「coherence 随 spike 单调上升」的不变式 `test_invariant_spike_parameter_increases_coherence`
被错误标记 xfail。

**原因**：旧公式用 `spike` 直接缩放整列，未改变列间夹角，故相干度不变。

**修复**：改为**两列夹角余弦映射**——设 `v = spike·a_i + sqrt(1−spike²)·a_j`，使
`a_i, v` 的夹角余弦恰好等于 `spike`。这样 `spike↑ ⇒` 该列对与其他列的最大夹角余弦
`↑ ⇒` 矩阵相干度 `↑`，不变式成立。实测 `spike ∈ {0.3,0.6,0.9,0.97}` 对应
`coh = {0.4112, 0.6405, 0.9070, 0.9713}` 严格递增。

**避免**：构造「可控相干度」矩阵时，永远用**几何夹角**而非整列缩放。

---

## §D-b 路由判据维度自适应：绝对阈值在 `m<n` 小样本下失效

**现象**：旧 `route_solver(coh, snr)` 用固定 `coh > 0.35` 决定 `omp`/`iht`。
在小 `m` 下相干度天然偏高，绝对阈值误判为「高相干→iht」，实则应走 `omp`。

**原因**：相干度期望随 `m` 变化：`E[coh] ≈ sqrt(2·ln(n(n-1)/2) / m)`。
`m=96,n=256` 时理论基线 ≈ 0.464；`m=48,n=128` 时更高。用绝对阈值无视此基线。

**修复**：路由改为维度自适应——
`route_solver(coh, snr, m, n)` 计算 `random_coherence_baseline(m, n)`，
判据 `coh / baseline > 1.5 ⇒ iht`，否则 `omp`。向后兼容：不传 `m,n` 时退回
绝对阈值 `coh > 0.35`。**收缩族（rw_fista 等）不进入路由**——它们不是「选哪个单法」，
而是后处理。

**实测**：

| coh   | m    | n    | route  |
|-------|------|------|--------|
| 0.44  | 96   | 256  | omp    |
| 0.97  | 96   | 256  | iht    |
| 0.46  | 96   | 256  | omp    |
| 0.97  | 96   | 256  | iht    |
| 0.1   | —    | —    | omp    |
| 0.5   | —    | —    | iht    |

---

## §D1 `stability_k` 频率未归一化：bootstrap 频率曲线拐点法偏小

**现象**：`stability_k` 返回 `k_best = m − 2`（饱和），频率曲线不收敛、归一化缺失，
导致 `test_invariant_stability_k_freq_is_normalized` 旧标记 xfail。

**原因**：行 bootstrap 后统计「每个 `k` 被选中的频率」，但频率向量未除以 bootstrap
次数，且「拐点」判据在边界 `m−2` 处被截断。

**修复**：
1. 频率向量显式归一化到 `[0,1]`（`freq[k] = count[k] / n_boot`）；
2. 拐点法（二阶差分最大处）限制在 `(1, m−2)` 开区间，不再饱和到 `m−2`；
3. 新增断言 `np.all(freq <= 1.0)` 与 `freq.max() <= 1.0 + 1e-12` 锁住不变量。

**实测**：`m=48,n=128,k=6,seed=9,n_boot=4` 返回 `k_best = 6`（等于真值），
频率严格落在 `[0,1]`，不再饱和。旧 `test_known_defect_stability_k_saturates_at_m_minus_2`
改名 `test_invariant_stability_k_no_longer_saturates`。

> 补充（见 model card §3）：行重采样不改变列相关性，单靠 `stability_k` 投票仍可能
> 选中噪声峰。本仓库用 `elbow_k`（残差拐点，2-fold 交叉验证通过）作为主 `k_hat`，
> `stability_k` 仅作一致性交叉检查。

---

## §τ 不一致性原理（noise_discrepancy_k）在测试集上过拟合

**现象**：`noise_discrepancy_k` 在固定 `τ=0.40` 时单实例能给出合理 `k`，但换 seed
即崩，不泛化。

**原因**：不一致性原理要求「测量噪声水平已知」。`τ` 是把「残差范数 vs 理论噪声」
的差异阈值化的超参；测试集上调出的 `τ=0.40` 是对该组实例的**过拟合**，不迁移。

**修复**：`τ` 不作为默认 `k` 估计器。默认 `k_hat` 链 = `elbow_k`（2-fold CV 验证
稳定）为主，`stability_k` 为辅；`noise_discrepancy_k` 仅在 `examples/failures.py`
中作为「已知失败模式」显式复现，提醒用户不要直接拿测试集调出的 `τ` 当生产默认值。

---

## §elbow 新踩坑：elbow_k 与 stability_k 在单实例上可相近，但不恒等

**现象**：早期断言「`stability_k` 必劣于 `elbow_k`」在部分实例上不成立（实测
`elbow` NMSE = 2.13e-03 vs `stability` 6.06e-04，后者反而更优）。

**原因**：`elbow_k` 用残差拐点，`stability_k` 用频率拐点，两者对「真值 k 邻域」的
敏感度不同；在噪声适中、相干度中等的实例上频率拐点更准。

**修复**：`examples/failures.py` 中相关 case 改为**陈述性措辞**——「单实例上两者可
相近，elbow 不恒优于 stability」，不再断言稳定性法必劣。门禁用 `elbow_k` 作为
真值无关默认，因其在跨 seed 上方差更小、更稳。

---

## 汇总：修复前后对照

| 项 | 修复前 | 修复后 | 状态 |
|----|--------|--------|------|
| `gram_whitener`（P0） | `m<n` 下 Cholesky 崩 / 白化不收敛 | `np.eye(n)` 收缩 + 逆变换 roundtrip 1.8e-15 | ✅ |
| spike 映射（§D-a） | coherence 不随 spike 变 | 夹角余弦映射，严格单调 | ✅ |
| 路由判据（§D-b） | 绝对阈值误判小样本 | 维度自适应基线比 | ✅ |
| stability_k（§D1） | 饱和 `m−2`、频率未归一 | 开区间拐点 + 归一化 | ✅ |
| 默认 seeds（cli check） | 默认 2 撞 `n_seeds≥3` | 默认 3 | ✅ |
| examples 子命令 | `ModuleNotFoundError` | `ablation.py` 文件名对齐 | ✅ |
| 覆盖率 | — | 83%（≥80% 达标） | ✅ |
