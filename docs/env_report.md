# SparseForge 环境报告（Phase 1）

> 作者：晨星
> 报告生成阶段：Phase 1（环境 + 骨架），不含算法主逻辑

## 1. 环境前提

| 项 | 实测值 |
| --- | --- |
| OS | Windows 11（win_amd64） |
| 托管 Python | `3.13.14 (main, Jun 11 2026, 04:04:46) [MSC v.1944 64 bit (AMD64)]` |
| 托管解释器路径 | `C:/Users/Administrator/.workbuddy/binaries/python/versions/3.13.12/python.exe` |
| pip（托管） | 26.2.1 |
| git | 2.55.0.windows.3 |
| 硬件约束 | 无 GPU；**禁止任何 C/C++ 源码编译** |
| 代理环境变量 | `http_proxy` / `https_proxy` = `http://127.0.0.1:31859`（沙箱代理，偶发 502） |

---

## 2. 轮子先验矩阵（win_amd64 + cp313）

探测命令（强制 binary-only，拿不到即判不可用，不降级到源码编译）：

```bash
pip download --no-deps --only-binary=:all: \
  --platform win_amd64 --python-version 3.13 --implementation cp \
  -d <tmpdir> <pkg> --proxy "" --index-url https://pypi.org/simple
```

| 包 | 探测版本 | 解析到的 wheel | wheel 可用性 | 结论 |
| --- | --- | --- | --- | --- |
| numpy | 2.5.3 | `numpy-2.5.3-cp313-cp313-win_amd64.whl` | ✅ | **Tier-1 必需**，原生 cp313 二进制 |
| scipy | 1.18.1 | `scipy-1.18.1-cp313-cp313-win_amd64.whl` | ✅ | **必需**，原生 cp313 二进制 |
| scikit-learn | 1.9.1 | `scikit_learn-1.9.1-cp313-cp313-win_amd64.whl` | ✅ | **Tier-0 可选后端**，原生 cp313 二进制 |
| pytest | 9.1.1 | `pytest-9.1.1-py3-none-any.whl` | ✅ | 必需（测试），纯 Python |
| pytest-cov | 7.1.0 | `pytest_cov-7.1.0-py3-none-any.whl` | ✅ | 必需（覆盖率），纯 Python |
| ruff | **0.16.10（钉版本）** | `ruff-0.16.10-py3-none-win_amd64.whl` | ✅ | 必需（lint），win_amd64 二进制 |
| sporco | 0.2.2.post1 | `sporco-0.2.2.post1-py3-none-any.whl` | ✅ | **可选**，纯 Python，无需编译 |
| pylops | 2.8.0 | `pylops-2.8.0-py3-none-any.whl` | ✅ | **可选**，纯 Python，无需编译 |

**汇总：8 / 8 全部有预编译 wheel，零编译、零阻塞。**

补充说明：

- `sporco` / `pylops` 均为 `py3-none-any` 纯 Python wheel，被判为**可用但不必装**。二者是学术算法参考实现（ADMM / 卷积稀疏编码、线性算子），本仓设计为 numpy Tier-1 自研 + sklearn Tier-0 可选后端，**不引入这两个依赖**，避免功能重复与依赖膨胀。若后续需做算法对照实验再按需加装。
- 黑名单项（lancedb / llama-cpp-python / torch-geometric / faiss-gpu 等需 C 编译重型包）未在本次探测中出现，全部规避。
- PyPI 直连可达：使用 `--proxy ""` + `--index-url https://pypi.org/simple` 成功，**无需切换 tuna 镜像**。

---

## 3. venv 与已装依赖

### 3.1 venv

| 项 | 值 |
| --- | --- |
| 状态 | ✅ 新建成功（`python -m venv`，未被沙箱拦截） |
| 解释器绝对路径 | `C:/Users/Administrator/.workbuddy/binaries/python/envs/sparseforge/Scripts/python.exe` |
| 版本 | 3.13.14 |
| `sys.prefix` | `C:\Users\Administrator\.workbuddy\binaries\python\envs\sparseforge`（已确认隔离） |

> Windows 下解释器在 `Scripts/`，**不是** `bin/`。

### 3.2 安装命令

```bash
"<venv>/Scripts/python.exe" -m pip install --only-binary=:all: \
  --proxy "" --index-url https://pypi.org/simple \
  numpy scipy scikit-learn pytest pytest-cov "ruff==0.16.10"
```

`Successfully installed ...`（16 个包，全部来自 wheel）。

### 3.3 已装依赖清单（`pip freeze`）

| 包 | 版本 | 来源 |
| --- | --- | --- |
| cloudpickle | 3.1.2 | 传递依赖（joblib） |
| colorama | 0.4.6 | 传递依赖（pytest） |
| coverage | 7.16.2 | 传递依赖（pytest-cov） |
| iniconfig | 2.3.0 | 传递依赖（pytest） |
| joblib | 1.6.0 | 传递依赖（scikit-learn） |
| narwhals | 2.26.0 | 传递依赖（scikit-learn） |
| **numpy** | **2.5.3** | 直接依赖（Tier-1） |
| packaging | 26.3 | 传递依赖（pytest） |
| pip | 26.1.2 | 环境自带 |
| pluggy | 1.6.0 | 传递依赖（pytest） |
| Pygments | 2.21.0 | 传递依赖（pytest） |
| **pytest** | **9.1.1** | 直接依赖 |
| **pytest-cov** | **7.1.0** | 直接依赖 |
| **ruff** | **0.16.10** | 直接依赖（已钉版本） |
| **scikit-learn** | **1.9.1** | 直接依赖（Tier-0 可选后端） |
| **scipy** | **1.18.1** | 直接依赖 |
| threadpoolctl | 3.7.0 | 传递依赖（scikit-learn） |

直接依赖一行式锁定：

```
numpy==2.5.3 scipy==1.18.1 scikit-learn==1.9.1 pytest==9.1.1 pytest-cov==7.1.0 ruff==0.16.10
```

### 3.4 导入与功能验证证据

```
=== import check ===
  numpy        OK    2.5.3
  scipy        OK    1.18.1
  sklearn      OK    1.9.1
  pytest       OK    9.1.1
  pytest_cov   OK    7.1.0

=== ruff ===
  <venv>/Scripts/ruff.exe --version   -> ruff 0.16.10
  <venv>/Scripts/python.exe -m ruff    -> ruff 0.16.10

=== 压缩感知功能冒烟（40x120 随机高斯测量矩阵，3-sparse 信号）===
  scipy version: 1.18.1
  sklearn version: 1.9.1
  OMP relative recovery error: 4.361e-16
  sparse matrix ok: (3, 3)

=== lint ===
  <venv>/Scripts/ruff.exe check .   -> All checks passed!
```

`4.361e-16` 已达浮点双精度极限，说明 numpy / scipy / sklearn 三者协同链路在本机完全可用，压缩感知基线（OMP）可直接跑通。

### 3.5 ⚠️ 已知风险：ruff 被 PATH 影子化

| 检查项 | 状态 | 证据 | 处置 |
| --- | --- | --- | --- |
| venv 内 ruff 版本 | ✅ | `<venv>/Scripts/ruff.exe --version` → `0.16.10` | 正确钉版 |
| PATH 上裸 `ruff` | ⚠️ | `ruff --version` → `0.16.9`；`which ruff` → `...\envs\default\Scripts\ruff.EXE` | 命中了 **default** venv 的旧版 |

**处置：所有 lint 调用必须走显式路径或 `python -m ruff`，禁止裸调 `ruff`。** CI 中同样使用 `"$VENV/Scripts/python.exe" -m ruff`。

---

## 4. 骨架目录树

仓库根目录 = `C:/Users/Administrator/WorkBuddy/2026-10-05-03-16-42/sparseforge/`
（**已核对：不存在 `sparseforge/sparseforge/` 嵌套**）

```
sparseforge/                      <- 仓库根
├── .github/
│   └── workflows/
│       └── .gitkeep              # CI 占位（Phase 2 填 ci.yml）
├── .gitignore
├── cli.py                        # 命令行入口占位
├── core/                         # 核心基础设施层
│   └── __init__.py
├── data/                         # 数据生成与加载层
│   └── __init__.py
├── cs/                           # 压缩感知核心算法层
│   └── __init__.py
├── preprocess/                   # 预处理与特征工程层
│   └── __init__.py
├── hpo/                          # 超参数优化层
│   └── __init__.py
├── training/                     # 训练/拟合层
│   └── __init__.py
├── eval/                         # 评测层
│   └── __init__.py
├── pipeline/                     # 端到端流水线编排层
│   └── __init__.py
├── examples/
│   └── README.md                 # 示例占位说明
├── tests/
│   └── README.md                 # 测试占位说明
└── docs/
    └── env_report.md             # 本文档
```

每个 `__init__.py` 与 `cli.py` 仅含中文模块 docstring（标注分层职责 + 作者晨星），**无任何算法实现**，符合 Phase 1 只搭骨架的要求。

分层数据流约定（Phase 2 实现时遵守）：

```
data → preprocess → cs → (hpo / training) → eval
                    ↑
                 pipeline 统一编排 + core 提供公共基础件
```

---

## 5. 推送通道探测结论

| 通道 | 状态 | 证据 | 处置 |
| --- | --- | --- | --- |
| `gh api user` | ✅ | 返回 `login: CJX0712`，`name: 晨星`，`id: 328722400` | 可用 |
| gh token scopes | ✅ | `delete_repo, gist, read:org, repo, user, workflow` | 含 `repo` / `workflow`，足够 push 与发 Release |
| git 全局署名 | ✅ | `晨星 <CJX0712@users.noreply.github.com>` | 与要求一致 |
| git 凭据链 | ✅ | `credential.https://github.com.helper = !'...\gh.exe' auth git-credential` | `git push` 可自动认证 |
| git smart-HTTP (https) | ✅ | `git ls-remote https://github.com/CJX0712/pgmforge.git HEAD` → 返回 SHA，3/3 连续成功 | **主通道** |
| SSH (`git@github.com`) | ❌ | `Permission denied (publickey)` | 无 SSH key，**不可用、不考虑** |
| 沙箱代理干扰 | ⚠️ | `gh_push.py --check` 曾报 `CONNECT tunnel failed, response 502`；随后直连重试 3/3 成功 | 代理偶发抖动，非永久失败 |
| 目标仓库 `CJX0712/sparseforge` | ⚠️ | `gh repo view` → GraphQL `Could not resolve`；`curl api` → `404` | **尚未创建**，push 前须先创建 |

### 5.1 结论与降级策略

后续推送按**三级降级**执行：

1. **L1（首选）：原生 `git push` over HTTPS**
   凭据链已就绪（`gh auth git-credential`），实测 https 智能协议连通。
   若遇代理 502，用 `git -c http.proxy= -c https.proxy= push ...` 绕过代理重试一次。

2. **L2（兜底）：`C:/Users/Administrator/.workbuddy/skills/random-ai-system-delivery/scripts/gh_push.py`**
   脚本存在（12800 字节），用法：

   ```
   usage: gh_push.py [-h] --repo REPO [--local LOCAL] [--owner OWNER]
                     [--description DESCRIPTION] [--tag TAG] [--release RELEASE]
                     [--check] [--force-api]
   ```
   
   - `--check`：只探通道，不推送（本次已用它探测）
   - `--force-api`：跳过 L1，直接走 GitHub API 提交
   - 支持 `--tag` / `--release`，可一步打 tag 与发 Release

3. **L3：GitHub Contents API 逐文件提交**（L2 内部已覆盖）

**前置动作：push 前必须先创建远端仓库 `CJX0712/sparseforge`**（当前 404）。
建议：`gh repo create CJX0712/sparseforge --public --description "..."` 或让 `gh_push.py` 自带创建逻辑兜底。

---

## 6. 已知限制与 Phase 2 待办

| 项 | 状态 | 说明 |
| --- | --- | --- |
| ruff PATH 影子化 | ⚠️ | 必须显式路径 / `python -m ruff` 调用 |
| 沙箱代理偶发 502 | ⚠️ | push / pip 失败时先重试一次并绕过代理 |
| 无 GPU | ⚠️ | 所有算法必须纯 CPU；不得引入 CUDA 依赖 |
| SSH 通道 | ❌ | 不可用，一律走 HTTPS |
| `sporco` / `pylops` | ⚠️ | wheel 可用但未安装；如需算法对照再按需加装 |
| `pyproject.toml` | ⬜ | Phase 1 未创建（超出骨架范围）。Phase 2 需要它来配置 pytest、ruff 规则与包元数据 |
| `.github/workflows/ci.yml` | ⬜ | 占位，Phase 2 填充（建议：ruff check + pytest + 覆盖率门禁） |
| 远端仓库 | ⬜ | 尚未创建，待主理人决定创建时机 |

---

## 7. 复现步骤（干净环境一键复现）

```bash
# 1) 建 venv
"C:/Users/Administrator/.workbuddy/binaries/python/versions/3.13.12/python.exe" \
  -m venv "C:/Users/Administrator/.workbuddy/binaries/python/envs/sparseforge"

# 2) 装依赖（强制 binary-only，禁止编译）
"C:/Users/Administrator/.workbuddy/binaries/python/envs/sparseforge/Scripts/python.exe" \
  -m pip install --only-binary=:all: --proxy "" --index-url https://pypi.org/simple \
  numpy==2.5.3 scipy==1.18.1 scikit-learn==1.9.1 \
  pytest==9.1.1 pytest-cov==7.1.0 ruff==0.16.10

# 3) 验证
"C:/Users/Administrator/.workbuddy/binaries/python/envs/sparseforge/Scripts/python.exe" \
  -c "import numpy,scipy,sklearn,pytest;print(numpy.__version__,scipy.__version__,sklearn.__version__,pytest.__version__)"
"C:/Users/Administrator/.workbuddy/binaries/python/envs/sparseforge/Scripts/python.exe" -m ruff --version

# 4) lint（注意：不要裸调 ruff，PATH 上是 0.16.9）
cd "C:/Users/Administrator/WorkBuddy/2026-10-05-03-16-42/sparseforge"
"C:/Users/Administrator/.workbuddy/binaries/python/envs/sparseforge/Scripts/python.exe" -m ruff check .
```

---

*作者：晨星*
