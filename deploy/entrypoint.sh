#!/usr/bin/env bash
set -euo pipefail

# SSO 跳转和回调由 didi_sso 在 sidecar 内处理；nginx 仅反向代理。
cp /app/deploy/nginx.conf /etc/nginx/conf.d/forecast.conf

# 删除默认站点, 避免与 forecast.conf 端口冲突
rm -f /etc/nginx/sites-enabled/default /etc/nginx/conf.d/default.conf 2>/dev/null || true

# 启动数据库迁移(幂等): 建表 + 补 owner_ldap 列 + 索引
python -m scripts.init_db || echo "[entrypoint] init_db skipped/failed (non-fatal)"

exec /usr/bin/supervisord -c /app/deploy/supervisord.conf
