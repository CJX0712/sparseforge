# SparseForge — Makefile
#
# 主要供 CI 使用（POSIX 命令）。Windows 用户在 Git Bash 下也可直接用。
#
# ⚠️ 重要约定（来自项目 SOP，硬门禁）：
#   lint 目标**不允许**出现任何吞掉退出码的写法（短路或、错误忽略等）。
#   ruff 报错必须让 CI 红，禁止降级放行。
#
# ⚠️ ruff 必须走 `python -m ruff`，不能用裸 `ruff`：
#   docs/env_report.md §3.5 实测 PATH 上的裸 ruff 是被 default venv
#   影子化的 0.16.9，而本项目钉的是 0.16.10。

PYTHON ?= python
RUFF   := $(PYTHON) -m ruff

.DEFAULT_GOAL := help
.PHONY: help install lint format format-check test cov demo check bench clean

help: ## 显示本可用法
	@echo "SparseForge — 可用目标："
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## 安装开发依赖（含 dev extras）
	$(PYTHON) -m pip install -e ".[dev]"

lint: ## ruff check —— 硬门禁，失败即退出非零
	$(RUFF) check .

format: ## ruff format 就地格式化
	$(RUFF) format .

format-check: ## ruff format --check（CI 用）
	$(RUFF) format --check .

test: ## 跑单元测试
	$(PYTHON) -m pytest

cov: ## 跑测试并生成覆盖率报告
	$(PYTHON) -m pytest --cov --cov-report=term-missing --cov-report=xml

demo: ## 端到端演示，产出 benchmark.json
	$(PYTHON) cli.py demo --out benchmark.json --seeds 3

bench: ## 跑基准并打印门禁明细
	$(PYTHON) cli.py bench --seeds 3

check: ## 确定性自检（同 seed 连跑两次逐位比对）
	$(PYTHON) cli.py check --seeds 3

clean: ## 清理缓存与构建产物
	rm -rf .ruff_cache .pytest_cache .coverage coverage.xml htmlcov build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +