# Forecast Lab 内网部署镜像 (app-only, 依赖外部 MySQL + 持久卷)
# 多阶段构建: builder 装依赖 -> runtime 跑 nginx + sso-sidecar + streamlit
#
# 公司内网构建提示:
#   - 基础镜像换成 hub.xiaojukeji.com/base/python:3.12-slim-bookworm
#   - uv 走内网源: ENV UV_INDEX_URL=https://pypi.intra.xiaojukeji.com/simple
#   - Apple Silicon 本地构建需 --platform linux/amd64 (OE 为 amd64)

# 基础镜像可通过 build-arg 覆盖:
#   生产内网: --build-arg BASE_IMAGE=hub.xiaojukeji.com/base/python:3.12-slim-bookworm
#   本地验证: 默认用本地已有的 python:3.12-slim-bookworm, 避免公网拉取超时
ARG BASE_IMAGE=python:3.12-slim-bookworm

# ── Stage 1: builder ──
FROM ${BASE_IMAGE} AS builder
 
 ENV UV_LINK_MODE=copy \
     UV_PYTHON_DOWNLOADS=never \
     UV_INDEX_URL=https://pypi.intra.xiaojukeji.com/simple \
     PIP_INDEX_URL=https://pypi.intra.xiaojukeji.com/simple

 RUN pip install --no-cache-dir -i https://pypi.intra.xiaojukeji.com/simple "uv>=0.5,<1"

 WORKDIR /app
 COPY pyproject.toml uv.lock ./
 # 内网环境去掉 pytorch-cpu 外网源(走内网 pypi 镜像装 CPU torch)
 RUN if [ -n "$UV_INDEX_URL" ]; then \
       sed -i '/\[tool.uv.sources\]/,/^$/{ /torch/d }' pyproject.toml && \
       sed -i '/pytorch-cpu/,+3d' pyproject.toml; \
     fi
 RUN uv sync --frozen --no-dev

# ── Stage 2: runtime ──
FROM ${BASE_IMAGE}

RUN apt-get update && apt-get install -y --no-install-recommends \
        nginx supervisor gettext-base \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    FORECAST_LAB_RUNTIME_DIR=/data

WORKDIR /app
COPY . /app

# 预下载模型权重(失败不阻断)
RUN python scripts/prefetch_models.py || true

RUN chmod +x deploy/entrypoint.sh

EXPOSE 80
VOLUME ["/data"]

ENTRYPOINT ["/app/deploy/entrypoint.sh"]
