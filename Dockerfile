# Forecast Lab 内网部署镜像 (app-only, 依赖外部 MySQL + 持久卷)
# 多阶段构建: builder 装依赖 -> runtime 跑 nginx + sso-sidecar + streamlit
#
# 公司内网构建提示:
#   - 基础镜像用内网镜像(如 hub.xiaojukeji.com/<ns>/python:3.12-slim-bookworm),
#     由 config/<env>/Dockerfile 指定, build.sh 自动替换进来
#   - PyPI 源: 不写死, 复用基础镜像内置的 pip 源(内网镜像已配好内网 PyPI);
#     该源同时喂给 uv(uv 不读 pip.conf)。也可 --build-arg PIP_INDEX_URL=... 覆盖
#   - Apple Silicon 本地构建需 --platform linux/amd64 (OE 为 amd64)

# 基础镜像可通过 build-arg 覆盖:
#   生产内网: --build-arg BASE_IMAGE=hub.xiaojukeji.com/base/python:3.12-slim-bookworm
#   本地验证: 默认用本地已有的 python:3.12-slim-bookworm, 避免公网拉取超时
ARG BASE_IMAGE=python:3.12-slim-bookworm

# ── Stage 1: builder ──
FROM ${BASE_IMAGE} AS builder

ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app
COPY pyproject.toml uv.lock ./

# PyPI 源可选地由 build-arg 覆盖; 缺省则复用基础镜像里 pip.conf 配好的内网源
ARG PIP_INDEX_URL=

# 内网构建: 解析出可用的 PyPI 源 -> 装 uv -> 去掉够不到的 pytorch-cpu 外网源 -> uv sync
# 必须放在同一个 RUN: INDEX 这个 shell 变量要从 pip 传递给 uv(uv 不读 pip.conf)
RUN set -eux; \
    INDEX="${PIP_INDEX_URL:-$(pip config get global.index-url 2>/dev/null || true)}"; \
    echo "Resolved PyPI index: ${INDEX:-<base image default>}"; \
    pip install --no-cache-dir ${INDEX:+-i "$INDEX"} "uv>=0.5,<1"; \
    sed -i '/^\[tool\.uv\.sources\]/,/^$/d; /^\[\[tool\.uv\.index\]\]/,/^$/d' pyproject.toml; \
    uv sync --frozen --no-dev ${INDEX:+--default-index "$INDEX"}

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
