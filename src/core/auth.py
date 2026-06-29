"""当前登录用户上下文。

生产环境: nginx 完成 SSO 鉴权后, 通过 HTTP 请求头 `X-SSO-User` 透传 LDAP,
Streamlit 从 `st.context.headers` 读取。
本地开发: 无 nginx, 回退到环境变量 `FORECAST_LAB_DEV_USER`(默认 "local-dev")。

约束: 所有 page / repository 调用一律走 `require_user()`/`current_user()`,
不要在业务代码里直接 grep header。LDAP 统一小写存储, 避免大小写歧义。
"""
from __future__ import annotations

import os

SSO_USER_HEADER = "X-SSO-User"
DEV_USER_ENV = "FORECAST_LAB_DEV_USER"
DEFAULT_DEV_USER = "local-dev"


def _normalize(ldap: str | None) -> str:
    return (ldap or "").strip().lower()


def _from_request_header() -> str:
    """从 Streamlit 请求头取 LDAP; 取不到返回空串(非 Streamlit 上下文 / header 缺失)。"""
    try:
        import streamlit as st  # 延迟 import, 便于在非 Streamlit 进程中复用本模块

        headers = st.context.headers  # Streamlit >= 1.37
        if headers:
            return _normalize(headers.get(SSO_USER_HEADER))
    except Exception:
        return ""
    return ""


def current_user() -> str:
    """返回当前用户 LDAP(小写)。

    优先级: 请求头 X-SSO-User > 环境变量 FORECAST_LAB_DEV_USER > "local-dev"。
    """
    ldap = _from_request_header()
    if ldap:
        return ldap
    return _normalize(os.environ.get(DEV_USER_ENV)) or DEFAULT_DEV_USER


def is_sso_authenticated() -> bool:
    """判断当前用户是否通过 SSO 登录(即请求头里有 X-SSO-User)。

    本地开发环境(走环境变量)返回 False, 让前端区分展示。
    """
    return bool(_from_request_header())


def require_user() -> str:
    """业务页面统一入口: 拿不到用户时显式中断页面渲染。

    生产中 nginx 已拦截未登录请求, 正常不会触发中断分支;
    这里作为最后一道兜底, 防止 header 异常时静默落到 default 用户。
    """
    ldap = current_user()
    if not ldap:
        try:
            import streamlit as st

            st.error("未识别到登录用户, 请刷新页面重新登录。")
            st.stop()
        except Exception as exc:
            raise PermissionError("missing current user (X-SSO-User)") from exc
    return ldap
