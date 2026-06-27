#!/usr/bin/env bash
set -euo pipefail

# nginx.conf 占位符默认值(生产)。本地集成可用 env 覆盖。
export SSO_APP_ID="${SSO_APP_ID:-}"
export SSO_LOGIN_BASE="${SSO_LOGIN_BASE:-http://mis.diditaxi.com.cn/auth/sso/login}"
export CALLBACK_SCHEME="${CALLBACK_SCHEME:-https}"

envsubst '${SSO_APP_ID} ${SSO_LOGIN_BASE} ${CALLBACK_SCHEME}' \
    < /app/deploy/nginx.conf > /etc/nginx/conf.d/forecast.conf

# 删除默认站点, 避免与 forecast.conf 端口冲突
rm -f /etc/nginx/sites-enabled/default /etc/nginx/conf.d/default.conf 2>/dev/null || true

# 启动数据库迁移(幂等): 建表 + 补 owner_ldap 列 + 索引
python -m scripts.init_db || echo "[entrypoint] init_db skipped/failed (non-fatal)"

exec /usr/bin/supervisord -c /app/deploy/supervisord.conf
