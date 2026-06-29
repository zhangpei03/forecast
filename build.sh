#!/bin/bash
set -e

APP_ENV=${APP_ENV:-test}
export APP_ENV

MODULE_NAME=forecast-lab

# ── 清理上次构建产物 ──
rm -rf output
rm -f $MODULE_NAME.zip

# ── 排除不需要进镜像的文件 ──
zip -r $MODULE_NAME.zip ./ \
  -x ".git/*" \
  -x ".venv/*" \
  -x ".pytest_cache/*" \
  -x ".ruff_cache/*" \
  -x "__pycache__/*" \
  -x "*.pyc" \
  -x ".DS_Store" \
  -x "output/*" \
  -x ".fpa_work/*" \
  -x "docs/*" \
  -x "tests/*" \
  -x "skills/*" \
  -x "*.log"

mkdir output
mv $MODULE_NAME.zip output/
cp deploy/entrypoint.sh output/

# OE 容器化部署配置
if [ "${APP_ENV}" = "dev" ] || [ "${APP_ENV}" = "test" ]; then
  cp Dockerfile output/
fi

cd output

# ── 解压到 output ──
unzip -o -d ./ ./$MODULE_NAME.zip
rm -f ./$MODULE_NAME.zip

# Copy environment-specific .env
if [ -f "config/${APP_ENV}/.env" ]; then
    cp config/${APP_ENV}/.env .env
    echo "Using config/${APP_ENV}/.env"
else
    echo "WARNING: No config/${APP_ENV}/.env found, using .env if present"
fi

echo "build APP_ENV=$APP_ENV"
