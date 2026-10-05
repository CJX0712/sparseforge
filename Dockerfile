# SparseForge — 纯 CLI 镜像（无长驻服务，故不暴露端口）
#
# 构建：
#   docker build -t sparseforge:0.1.0 .
# 运行自检：
#   docker run --rm sparseforge:0.1.0
# 跑基准：
#   docker run --rm sparseforge:0.1.0 bench --seeds 3
# 交互式 shell：
#   docker run --rm -it sparseforge:0.1.0 bash

# ---------------------------------------------------------------------------
# Stage 1: 依赖层（单独缓存，只有 requirements 变化才重建）
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS deps

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# 只装运行时依赖，保持镜像层小
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: 运行时镜像
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="SparseForge" \
      org.opencontainers.image.description="压缩感知/稀疏信号恢复系统（作者 晨星）" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/CJX0712/sparseforge"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# 非 root 用户（安全最佳实践）
RUN groupadd --system --gid 1001 sparseforge \
 && useradd  --system --uid 1001 --gid sparseforge --no-create-home sparseforge

# 先切到非 root 再拷贝源码，避免文件属主问题
COPY --chown=sparseforge:sparseforge core/    core/
COPY --chown=sparseforge:sparseforge cs/      cs/
COPY --chown=sparseforge:sparseforge data/    data/
COPY --chown=sparseforge:sparseforge preprocess/ preprocess/
COPY --chown=sparseforge:sparseforge hpo/     hpo/
COPY --chown=sparseforge:sparseforge training/ training/
COPY --chown=sparseforge:sparseforge eval/    eval/
COPY --chown=sparseforge:sparseforge pipeline/ pipeline/
COPY --chown=sparseforge:sparseforge cli.py   cli.py
COPY --chown=sparseforge:sparseforge pyproject.toml README.md LICENSE ./

USER sparseforge

# 默认动作：确定性自检（同 seed 连跑两次逐位比对）。
# 注意 cli.py check 的 --seeds 默认值 2 会触发 pipeline 的 >=3 约束，
# 因此这里显式给 --seeds 3。
CMD ["python", "cli.py", "check", "--seeds", "3"]