 #!/bin/bash
 set -e
 
 APP_ENV=${APP_ENV:-test}
 export APP_ENV
 
 echo "=== Forecast Lab build start APP_ENV=$APP_ENV ==="
 
 # ── 清理上次构建产物 ──
 rm -rf output
 mkdir -p output
 
 # ── 拷贝源码到 output(排除无关文件) ──
 rsync -a \
   --exclude='.git' \
   --exclude='.venv' \
   --exclude='.pytest_cache' \
   --exclude='.ruff_cache' \
   --exclude='__pycache__' \
   --exclude='*.pyc' \
   --exclude='.DS_Store' \
   --exclude='output' \
   --exclude='.fpa_work' \
   --exclude='docs' \
   --exclude='tests' \
   --exclude='skills' \
   --exclude='*.log' \
   --exclude='.env.example' \
   ./ output/
 
 # ── 确保 deploy 子目录完整(nginx/sidecar/supervisord) ──
 # rsync 已自动拷入, 这里显式确认关键文件存在
 for f in Dockerfile deploy/entrypoint.sh deploy/nginx.conf deploy/supervisord.conf deploy/sso_sidecar/main.py; do
   if [ ! -f "output/$f" ]; then
     echo "ERROR: missing output/$f" >&2
     exit 1
   fi
 done
 
 # Copy environment-specific .env
 if [ -d "config" ] && [ -f "config/${APP_ENV}/.env" ]; then
     cp "config/${APP_ENV}/.env" output/.env
     echo "Using config/${APP_ENV}/.env"
 else
     echo "No config/${APP_ENV}/.env, using .env if present"
 fi
 
 echo "=== build done, output/ contents ==="
 ls -la output/
