"""SSO sidecar backed by the company-standard :mod:`didi_sso` library.

The sidecar is the only public upstream for Streamlit.  ``SsoMiddleware``
owns the login callback and ticket validation flow; this module only forwards
an authenticated request and exposes the LDAP to Streamlit as ``X-SSO-User``.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from urllib.parse import parse_qs, urlencode, urljoin, urlsplit, urlunsplit

import httpx
import websockets
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware

from didi_sso import (
    McpAuthService,
    SsoMiddleware,
    SsoService,
    User,
    create_sso_router,
    get_user,
)

load_dotenv(override=False)
logger = logging.getLogger("forecast_sso_sidecar")


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _env_bool(key: str, default: bool = False) -> bool:
    return _env(key, str(default)).lower() in {"true", "1", "yes", "on"}


SSO_ENABLED = _env_bool("SSO_ENABLED", True)
SSO_APP_ID = _env("SSO_APP_ID")
SSO_APP_KEY = _env("SSO_APP_KEY")
SSO_HOST = _env("SSO_HOST", "https://mis.diditaxi.com.cn")
SSO_DOMAIN = _env("SSO_DOMAIN")
SSO_LOGIN_PATH = _env("SSO_LOGIN_PATH", "/auth/sso/login")
SSO_LOGOUT_PATH = _env("SSO_LOGOUT_PATH", "/auth/ldap/logout")
SSO_CHECK_TICKET_PATH = _env("SSO_CHECK_TICKET_PATH", "/auth/sso/api/check_ticket")
SSO_CHECK_CODE_PATH = _env("SSO_CHECK_CODE_PATH", "/auth/sso/api/check_code")
SSO_USER_INDEX_PATH = _env("SSO_USER_INDEX_PATH", "/auth/api/user/index")
UPM_CHECK_USER_TICKET_PATH = _env("UPM_CHECK_USER_TICKET_PATH", "/auth/sso/api/get_user_by_ticket")
SSO_MOCK = _env_bool("SSO_MOCK", False)
SSO_DEFAULT_LDAP = _env("SSO_DEFAULT_LDAP")
SSO_DEFAULT_USER_NAME = _env("SSO_DEFAULT_USER_NAME")
MCP_APP_ID = _env("MCP_APP_ID")
MCP_SECRET_KEY = _env("MCP_SECRET_KEY")
APP_ENV = _env("APP_ENV", "prod")
STREAMLIT_URL = _env("STREAMLIT_URL", "http://127.0.0.1:8501")
PUBLIC_BASE_URL = _env("PUBLIC_BASE_URL").rstrip("/")
SSO_CALLBACK_PATH = "/sso/callback"

sso_service = SsoService(
    app_id=SSO_APP_ID,
    app_key=SSO_APP_KEY,
    sso_host=SSO_HOST,
    domain=SSO_DOMAIN,
    login_path=SSO_LOGIN_PATH,
    logout_path=SSO_LOGOUT_PATH,
    check_ticket_path=SSO_CHECK_TICKET_PATH,
    check_code_path=SSO_CHECK_CODE_PATH,
    mock=SSO_MOCK,
    user_index_path=SSO_USER_INDEX_PATH,
    upm_check_user_ticket_path=UPM_CHECK_USER_TICKET_PATH,
)

def _sso_login_url(jump_to: str = "") -> str:
    """Build a browser login URL compatible with the SSO gateway.

    The ``jumpto`` is wrapped in our callback URL so UPM matches it as the
    registered callback.  The SSO gateway then generates a ``code`` parameter
    and redirects to ``/sso/callback?code=...&jumpto=<target>``.
    SsoMiddleware (Priority 2) exchanges the code for a ticket, writes the
    cookie, and the callback endpoint redirects to the original target.
    """
    params = {"app_id": SSO_APP_ID, "version": "1.0"}
    if jump_to:
        params["jumpto"] = _callback_jump_to(jump_to)
    return f"{SSO_HOST}{SSO_LOGIN_PATH}?{urlencode(params)}"


sso_service.get_login_url = _sso_login_url

mcp_auth = McpAuthService(mcp_app_id=MCP_APP_ID, mcp_secret_key=MCP_SECRET_KEY)

default_user = (
    User(ldap=SSO_DEFAULT_LDAP, user_name=SSO_DEFAULT_USER_NAME)
    if SSO_DEFAULT_LDAP and APP_ENV != "prod"
    else None
)


def _public_request_url(request: Request) -> str:
    """Build the externally visible URL for SSO's post-login redirect."""
    if PUBLIC_BASE_URL:
        public = urlsplit(PUBLIC_BASE_URL)
        return urlunsplit((public.scheme, public.netloc, request.url.path, request.url.query, ""))
    scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("x-forwarded-host") or request.headers.get("host", request.url.netloc)
    return urlunsplit((scheme, host, request.url.path, request.url.query, ""))


def _unwrap_callback_jump_to(jump_to: str) -> str:
    """Avoid recursively using the SSO callback URL as the next login target."""
    current = jump_to
    for _ in range(5):
        parsed = urlsplit(current)
        if parsed.path != SSO_CALLBACK_PATH:
            return current
        nested = parse_qs(parsed.query).get("jumpto", [""])[0]
        if not nested or nested == current:
            break
        current = nested
    return "/"


def _public_base_from_target(target: str) -> str:
    """Return the externally visible origin used to build the SSO callback URL."""
    if PUBLIC_BASE_URL:
        return PUBLIC_BASE_URL
    parsed = urlsplit(target)
    if parsed.scheme and parsed.netloc:
        return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
    return ""


def _callback_jump_to(jump_to: str) -> str:
    """Wrap the original target in our callback so UPM can match the callback URL.

    UPM matches the login ``jumpto`` after stripping query parameters. Therefore
    the SSO-facing target must be ``/sso/callback``; the real page is carried as
    the callback's own ``jumpto`` parameter.
    """
    target = _unwrap_callback_jump_to(jump_to)
    base_url = _public_base_from_target(target).rstrip("/")
    callback_url = f"{base_url}{SSO_CALLBACK_PATH}" if base_url else SSO_CALLBACK_PATH
    return f"{callback_url}?{urlencode({'jumpto': target})}"


class BrowserLoginRedirectMiddleware(BaseHTTPMiddleware):
    """Turn the SDK's browser-facing 401 JSON response into an SSO redirect."""

    def __init__(self, app, *, sso_service: SsoService) -> None:
        super().__init__(app)
        self.sso_service = sso_service

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        accepts_html = "text/html" in request.headers.get("accept", "")
        if (
            request.method in {"GET", "HEAD"}
            and response.status_code == 401
            and accepts_html
        ):
            if request.url.path == SSO_CALLBACK_PATH:
                # Callback 401: no valid code or ticket. Extract original
                # target and retry login once to avoid infinite loops.
                if request.query_params.get("_retry"):
                    return response
                jump_to = request.query_params.get("jumpto") or "/"
                target = _unwrap_callback_jump_to(jump_to)
            else:
                target = _public_request_url(request)
            login_url = _sso_login_url(target)
            return RedirectResponse(url=login_url, status_code=302)
        return response


app = FastAPI(title="forecast-sso-sidecar")


async def _callback_params(request: Request) -> dict[str, str]:
    """Read callback parameters from query, form-encoded, or JSON requests."""
    params = {key: value for key, value in request.query_params.items()}
    if request.method in {"GET", "HEAD"}:
        return params

    body = await request.body()
    if not body:
        return params

    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            return params
        if isinstance(payload, dict):
            params.update({str(key): str(value) for key, value in payload.items()})
        return params

    for key, values in parse_qs(body.decode("utf-8")).items():
        if values:
            params[key] = values[-1]
    return params


def _log_callback_state(request: Request, params: dict[str, str], stage: str) -> None:
    """Log callback diagnostics without leaking ticket/code/token values."""
    interesting = ["code", "ticket", "token", "access_token", "jumpto"]
    present = [key for key in interesting if params.get(key)]
    logger.warning(
        "sso_callback stage=%s method=%s path=%s keys=%s present=%s content_type=%s",
        stage,
        request.method,
        request.url.path,
        sorted(params.keys()),
        present,
        request.headers.get("content-type", ""),
    )


@app.api_route(SSO_CALLBACK_PATH, methods=["GET", "POST"])
async def sso_callback(request: Request) -> Response:
    """Redirect to the original target page after SsoMiddleware has exchanged
    the OAuth code for a ticket and written the cookie."""
    jump_to = request.query_params.get("jumpto") or "/"
    target = _unwrap_callback_jump_to(jump_to)
    logger.info("sso_callback redirecting to %s", target[:200])
    return RedirectResponse(url=target, status_code=302)


@app.get("/sso/logout")
async def sso_logout(request: Request) -> Response:
    """Explicit logout entry point for Streamlit sidebar link.

    Returns a redirect to the SSO gateway logout page and clears the local
    ticket cookie.
    """
    referer = request.headers.get("referer", "")
    logout_url = sso_service.get_logout_url(referer)
    response = RedirectResponse(url=logout_url, status_code=302)
    sso_service.remove_ticket_cookie(response)
    return response


@app.get("/sso/login")
async def sso_login(request: Request) -> Response:
    """Explicit login entry point for Streamlit sidebar link.

    Redirects the browser to the SSO gateway login page.  The jumpto is
    wrapped in the callback URL so the gateway generates a code parameter;
    SsoMiddleware exchanges the code for a ticket on the callback path.
    """
    target = _public_request_url(request)
    login_url = _sso_login_url(target)
    return RedirectResponse(url=login_url, status_code=302)


@app.get("/sso/debug-cookie")
async def debug_sso_cookie(request: Request) -> dict[str, object]:
    """Report whether the browser is sending the local SSO cookie."""
    cookie_name = sso_service._ticket_cookie_name()
    return {
        "cookie_name": cookie_name,
        "has_cookie": bool(request.cookies.get(cookie_name)),
        "all_cookie_names": sorted(request.cookies.keys()),
    }


@app.get("/sso/debug-login-url")
async def debug_login_url(request: Request) -> dict[str, str]:
    """Report the login URL shape without exposing secrets."""
    public_url = _public_request_url(request)
    return {"public_url": public_url, "login_url": sso_service.get_login_url(public_url)}


app.add_middleware(
    SsoMiddleware,
    sso_service=sso_service,
    mcp_auth=mcp_auth,
    enabled=SSO_ENABLED,
    default_user=default_user,
    skip_paths=frozenset(["/health", "/sso/login", "/sso/logout", "/openapi.json", "/docs", "/redoc"]),
)
# Add after SsoMiddleware: Starlette executes the latest registered middleware first,
# so this outer layer can convert didi_sso's browser 401 response into a redirect.
app.add_middleware(BrowserLoginRedirectMiddleware, sso_service=sso_service)
app.include_router(create_sso_router(sso_service, prefix="/sso"))


@app.get("/health")
async def health() -> dict[str, str]:
    """Unauthenticated liveness endpoint for the container platform."""
    return {"status": "ok"}


def _get_user_ldap() -> str:
    """Read the user from the ``didi_sso`` request ContextVar."""
    user = get_user()
    if isinstance(user, User):
        return str(user.ldap).strip().lower()
    return ""


async def _get_websocket_user(websocket: WebSocket) -> User | None:
    """Authenticate the WebSocket, which ``BaseHTTPMiddleware`` does not see."""
    if not SSO_ENABLED:
        return default_user

    ticket = sso_service.get_ticket(dict(websocket.cookies))
    if not ticket or not await sso_service.check_ticket(ticket):
        return None
    return await sso_service.get_user_info(ticket)


def _streamlit_websocket_url(query: str) -> str:
    parsed = urlsplit(STREAMLIT_URL)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    path = f"{parsed.path.rstrip('/')}/_stcore/stream"
    return urlunsplit((scheme, parsed.netloc, path, query, ""))


@app.api_route(
    "/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]
)
async def proxy_http(request: Request, path: str) -> Response:
    """Forward an authenticated HTTP request to Streamlit.

    Any incoming ``X-SSO-User`` is removed before the trusted middleware value
    is added, so callers cannot impersonate another LDAP account.
    """
    target_url = urljoin(f"{STREAMLIT_URL.rstrip('/')}/", path)
    if request.url.query:
        target_url = f"{target_url}?{request.url.query}"

    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in {"host", "x-sso-user"}
    }
    headers["X-SSO-User"] = _get_user_ldap()

    async with httpx.AsyncClient(timeout=600.0) as client:
        upstream = await client.request(
            method=request.method,
            url=target_url,
            headers=headers,
            content=await request.body(),
        )

    response = Response(content=upstream.content, status_code=upstream.status_code)
    excluded_headers = {"connection", "content-encoding", "content-length", "transfer-encoding"}
    for key, value in upstream.headers.multi_items():
        if key.lower() not in excluded_headers:
            response.headers.append(key, value)
    return response


@app.websocket("/_stcore/stream")
async def proxy_websocket(websocket: WebSocket) -> None:
    """Bidirectionally proxy Streamlit's text and binary WebSocket frames."""
    user = await _get_websocket_user(websocket)
    if user is None:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    try:
        async with websockets.connect(
            _streamlit_websocket_url(websocket.url.query),
            additional_headers={"X-SSO-User": user.ldap.strip().lower()},
        ) as upstream:

            async def client_to_server() -> None:
                while True:
                    message = await websocket.receive()
                    if message["type"] == "websocket.disconnect":
                        return
                    if message.get("text") is not None:
                        await upstream.send(message["text"])
                    elif message.get("bytes") is not None:
                        await upstream.send(message["bytes"])

            async def server_to_client() -> None:
                while True:
                    message = await upstream.recv()
                    if isinstance(message, bytes):
                        await websocket.send_bytes(message)
                    else:
                        await websocket.send_text(message)

            await asyncio.gather(client_to_server(), server_to_client())
    except WebSocketDisconnect:
        pass
    finally:
        if websocket.client_state.name != "DISCONNECTED":
            await websocket.close()
