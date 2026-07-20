ARG BASE_IMAGE=python:3.12-slim-bookworm

FROM ${BASE_IMAGE}

RUN apt-get update && apt-get install -y --no-install-recommends \
        nginx supervisor gettext-base \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    FORECAST_LAB_RUNTIME_DIR=/data \
    APP_ENV=prod \
    SSO_ENABLED=true \
    SSO_APP_ID=2102571 \
    SSO_HOST=https://mis.diditaxi.com.cn \
    PUBLIC_BASE_URL=https://forecast.intra.xiaojukeji.com \
    SSO_DOMAIN= \
    SSO_LOGIN_PATH=/auth/sso/login \
    SSO_LOGOUT_PATH=/auth/ldap/logout \
    SSO_CHECK_TICKET_PATH=/auth/sso/api/check_ticket \
    SSO_CHECK_CODE_PATH=/auth/sso/api/check_code \
    SSO_USER_INDEX_PATH=/auth/api/user/index \
    UPM_CHECK_USER_TICKET_PATH=/auth/sso/api/get_user_by_ticket \
    SSO_MOCK=false

WORKDIR /app
COPY . /app

RUN python scripts/prefetch_models.py || true
RUN chmod +x deploy/entrypoint.sh

EXPOSE 80
VOLUME ["/data"]
ENTRYPOINT ["/app/deploy/entrypoint.sh"]
