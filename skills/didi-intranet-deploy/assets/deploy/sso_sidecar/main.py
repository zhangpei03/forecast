"""滴滴 SSO 鉴权 sidecar (FastAPI)。

由 nginx auth_request 调用, 不直接对外暴露。三个路由对应 mis.diditaxi.com.cn 三接口:
  - GET /sso/check    : nginx 子请求, 校验 cookie 里的 ticket; 通过则 200 + X-SSO-User
  - GET /sso/callback : SSO 登录回跳, code -> ticket, 加密写 cookie, 302 回 jumpto
  - GET /sso/logout   : 清 cookie, 302 到统一登出

协议为类 CAS 的 code->ticket 模型 (非 OAuth2/OIDC)。
所有密钥 (APP_ID/APP_KEY/COOKIE_SECRET) 从环境变量读取, 不入仓。
"""
from __future__ import annotations

import os
from urllib.parse import quote

import httpx
from cryptography.fernet import Fernet, InvalidToken
from fastapi import FastAPI, Request, Response
from fastapi.responses import PlainTextResponse, RedirectResponse

APP_ID = os.environ.get("SSO_APP_ID", "")
APP_KEY = os.environ.get("SSO_APP_KEY", "")
COOKIE_SECRET = os.environ.get("SSO_COOKIE_SECRET", "")
COOKIE_NAME = os.environ.get("SSO_COOKIE_NAME", "fc_sso")
COOKIE_MAX_AGE = int(os.environ.get("SSO_COOKIE_MAX_AGE", "28800"))  # 8h
# 生产走 https, cookie 必须 Secure; 本地集成走 http 时设为 "0" 以便浏览器存 cookie
COOKIE_SECURE = os.environ.get("SSO_COOKIE_SECURE", "1") not in ("0", "false", "False")

LOGIN_URL = os.environ.get("SSO_LOGIN_URL", "http://mis.diditaxi.com.cn/auth/sso/login")
CHECK_CODE_URL = os.environ.get(
    "SSO_CHECK_CODE_URL", "http://mis.diditaxi.com.cn/auth/sso/api/check_code"
)
CHECK_TICKET_URL = os.environ.get(
    "SSO_CHECK_TICKET_URL", "http://mis.diditaxi.com.cn/auth/sso/api/check_ticket"
)
LOGOUT_URL = os.environ.get("SSO_LOGOUT_URL", "http://mis.diditaxi.com.cn/auth/ldap/logout")

# 本地/staging 用假身份打通链路, 不接真 SSO; 生产务必留空。
DEV_FAKE_USER = os.environ.get("SSO_DEV_FAKE_USER", "")

app = FastAPI(title="forecast-sso-sidecar")

_fernet = Fernet(COOKIE_SECRET.encode()) if COOKIE_SECRET else None


def _encrypt(value: str) -> str:
    if _fernet is None:
        return value
    return _fernet.encrypt(value.encode()).decode()


def _decrypt(token: str) -> str | None:
    if _fernet is None:
        return token
    try:
        return _fernet.decrypt(token.encode()).decode()
    except (InvalidToken, Exception):
        return None


# 不同 SSO 实现返回字段名略有差异, 这里做宽容匹配, 联调时一般不用改代码。
_SUCCESS_KEYS = ("errno", "errcode", "code", "ret", "status")
_USER_KEYS = ("username", "user", "userName", "empId", "ldap", "email", "loginName")
_TICKET_KEYS = ("ticket", "st", "token")


def _is_success(data: dict) -> bool:
    """成功标志兼容 errno/errcode/code/ret/status == 0 / "0" / "success" / true。"""
    for k in _SUCCESS_KEYS:
        if k in data:
            return data[k] in (0, "0", "success", "ok", "OK", True)
    # 没有任何状态字段时, 只要能取到用户名就当成功
    return True


def _pick(d: dict, keys: tuple[str, ...]) -> str | None:
    for k in keys:
        if isinstance(d, dict) and d.get(k):
            return str(d[k]).strip()
    return None


def _extract_user(data: dict) -> str | None:
    """从返回体(可能嵌在 data/result 里)提取用户名并小写化。email 取 @ 前缀。"""
    body = data.get("data") or data.get("result") or data
    user = _pick(body, _USER_KEYS) or _pick(data, _USER_KEYS)
    if not user:
        return None
    user = user.split("@")[0]  # email -> ldap 前缀
    return user.strip().lower() or None


def _extract_ticket(data: dict) -> str | None:
    body = data.get("data") or data.get("result") or data
    return _pick(body, _TICKET_KEYS) or _pick(data, _TICKET_KEYS)


async def _verify_ticket(ticket: str) -> str | None:
    """调 SSO check_ticket 校验 ticket, 返回 LDAP 用户名或 None。"""
    if DEV_FAKE_USER:
        return DEV_FAKE_USER
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                CHECK_TICKET_URL, params={"appid": APP_ID, "ticket": ticket}
            )
            data = resp.json()
    except Exception:
        return None
    if isinstance(data, dict) and _is_success(data):
        return _extract_user(data)
    return None


async def _exchange_code(code: str) -> tuple[str | None, str | None]:
    """调 SSO check_code 用 code 换 (ticket, username)。"""
    if DEV_FAKE_USER:
        return ("dev-ticket", DEV_FAKE_USER)
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                CHECK_CODE_URL, params={"appid": APP_ID, "appkey": APP_KEY, "code": code}
            )
            data = resp.json()
    except Exception:
        return (None, None)
    if isinstance(data, dict) and _is_success(data):
        return (_extract_ticket(data), _extract_user(data))
    return (None, None)


@app.get("/sso/check")
async def sso_check(request: Request) -> Response:
    """nginx auth_request 子请求: 通过返回 200 + X-SSO-User, 否则 401。"""
    cookie = request.cookies.get(COOKIE_NAME)
    if not cookie:
        return PlainTextResponse("no cookie", status_code=401)
    ticket = _decrypt(cookie)
    if not ticket:
        return PlainTextResponse("bad cookie", status_code=401)
    user = await _verify_ticket(ticket)
    if not user:
        return PlainTextResponse("invalid ticket", status_code=401)
    return Response(status_code=200, headers={"X-SSO-User": user})


@app.get("/sso/callback")
async def sso_callback(code: str = "", jumpto: str = "/") -> Response:
    """SSO 登录后回跳: code -> ticket, 加密写 cookie, 302 回业务页。"""
    if not code:
        return PlainTextResponse("missing code", status_code=400)
    ticket, user = await _exchange_code(code)
    if not ticket or not user:
        return PlainTextResponse("code exchange failed", status_code=401)
    resp = RedirectResponse(url=jumpto or "/", status_code=302)
    resp.set_cookie(
        key=COOKIE_NAME,
        value=_encrypt(ticket),
        max_age=COOKIE_MAX_AGE,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="lax",
        path="/",
    )
    return resp


@app.get("/sso/logout")
async def sso_logout() -> Response:
    resp = RedirectResponse(url=LOGOUT_URL, status_code=302)
    resp.delete_cookie(COOKIE_NAME, path="/")
    return resp


@app.get("/sso/login")
async def sso_login(request: Request) -> Response:
    """便捷登录入口: 拼 SSO 登录 URL 并 302。nginx 也可直接 302, 这里冗余兜底。"""
    host = request.headers.get("host", "")
    jumpto = quote(f"https://{host}/sso/callback", safe="")
    return RedirectResponse(url=f"{LOGIN_URL}?appid={APP_ID}&jumpto={jumpto}", status_code=302)
