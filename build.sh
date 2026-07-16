 #!/bin/bash
 set -ex
 
 APP_ENV=${APP_ENV:-test}
 export APP_ENV
 
 echo "=== Forecast Lab build start APP_ENV=$APP_ENV ==="
 echo "pwd=$(pwd)"
 echo "ls=$(ls)"
 
 # ── 清理上次构建产物 ──
 rm -rf output
 mkdir -p output
 
 # ── 拷贝源码到 output(排除无关文件, 只用 cp/find, 不依赖 rsync/zip) ──
 
 # 先拷贝顶层文件
 for f in app.py pyproject.toml uv.lock Dockerfile Makefile .env; do
   [ -f "$f" ] && cp "$f" output/
 done
 
 # 拷贝目录(排除 __pycache__ 和 .pyc)
 for d in src pages deploy scripts runtime sample_data .streamlit config vendor; do
   if [ -d "$d" ]; then
     mkdir -p "output/$d"
     find "$d" -type f ! -name '*.pyc' ! -path '*__pycache__*' -exec cp --parents {} output/ \;
   fi
 done
 
 # ── 校验关键文件 ──
 for f in Dockerfile deploy/entrypoint.sh deploy/nginx.conf deploy/supervisord.conf deploy/sso_sidecar/main.py; do
   if [ ! -f "output/$f" ]; then
     echo "ERROR: missing output/$f" >&2
     exit 1
   fi
 done
 
   # ── 替换 Dockerfile 基础镜像为内网镜像 ──
   if [ -f "config/${APP_ENV}/Dockerfile" ]; then
       INTERNAL_IMAGE=$(head -1 "config/${APP_ENV}/Dockerfile" | sed -E 's/^[[:space:]]*FROM[[:space:]]+//; s/[[:space:]]+$//')
       echo "Using internal base image: ${INTERNAL_IMAGE}"
       # 替换 ARG BASE_IMAGE 和所有 FROM 行 (容忍行首空白, 防止 ^锚点失配; 用 BRE 兼容 GNU/BSD sed)
       sed -i "s|^[[:space:]]*ARG BASE_IMAGE=.*|ARG BASE_IMAGE=${INTERNAL_IMAGE}|g" output/Dockerfile
       sed -i "s|^[[:space:]]*FROM \${BASE_IMAGE} AS builder.*|FROM ${INTERNAL_IMAGE} AS builder|g" output/Dockerfile
       sed -i "s|^[[:space:]]*FROM \${BASE_IMAGE}[[:space:]]*\$|FROM ${INTERNAL_IMAGE}|g" output/Dockerfile
       echo "Dockerfile after replacement:"
       grep -nE 'ARG BASE_IMAGE|^[[:space:]]*FROM' output/Dockerfile
   else
       echo "No config/${APP_ENV}/Dockerfile, using default Dockerfile"
   fi
 
  # Copy environment-specific .env
  if [ -d "config" ] && [ -f "config/${APP_ENV}/.env" ]; then
      cp "config/${APP_ENV}/.env" output/.env
      echo "Using config/${APP_ENV}/.env"
  else
      echo "No config/${APP_ENV}/.env, using .env if present"
  fi
 
  echo "=== build done ==="
  echo "output/ top-level:"
  ls output/
  echo "output/Dockerfile first line:"
  head -1 output/Dockerfile
