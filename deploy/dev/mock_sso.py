"""本地集成测试用的假 SSO 服务 (绝不用于生产)。

模拟 mis.diditaxi.com.cn 的三个接口, 但身份完全由 URL 参数决定, 方便用不同
用户名验证"多用户数据隔离":

  - GET /auth/sso/login?appid=&jumpto=   : 返回一个登录页, 让你选/输入用户名,
                                           然后 302 回 <jumpto>?code=code-<user>
  - GET /auth/sso/api/check_code?code=   : code-<user> -> {ticket: tkt-<user>, username:<user>}
  - GET /auth/sso/api/check_ticket?ticket=: tkt-<user> -> {username:<user>}
  - GET /auth/ldap/logout                : 简单返回已登出

ticket 里直接编码用户名 => 不同用户拿到不同 ticket => sidecar 据此识别不同身份。
"""
from __future__ import annotations

from urllib.parse import quote

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

app = FastAPI(title="mock-sso")

PRESET_USERS = ["alice", "bob", "parkerzhang"]


def _user_from_ticket(ticket: str) -> str | None:
    return ticket[len("tkt-"):] if ticket.startswith("tkt-") else None


def _user_from_code(code: str) -> str | None:
    return code[len("code-"):] if code.startswith("code-") else None


@app.get("/auth/sso/login", response_class=HTMLResponse)
def login(jumpto: str = "/", appid: str = "") -> HTMLResponse:
    buttons = "".join(
        f'<a class="btn" href="?__pick={u}&jumpto={quote(jumpto, safe="")}&appid={appid}">'
        f'以 <b>{u}</b> 登录</a>'
        for u in PRESET_USERS
    )
    html = f"""
    <html><head><meta charset="utf-8"><title>Mock SSO 登录</title>
    <style>
      body{{font-family:-apple-system,system-ui,sans-serif;max-width:520px;margin:60px auto;padding:0 20px;}}
      .btn{{display:block;padding:12px 16px;margin:10px 0;border:1px solid #ccc;border-radius:8px;
            text-decoration:none;color:#1a1a1a;background:#f7f8fa;}}
      .btn:hover{{background:#eef;}}
      form{{margin-top:20px;}} input{{padding:8px;width:60%;}} button{{padding:8px 14px;}}
      @media (prefers-color-scheme: dark){{body{{background:#111;color:#eee;}}.btn{{background:#222;border-color:#444;color:#eee;}}}}
    </style></head>
    <body>
      <h2>Mock SSO 登录</h2>
      <p>这是本地集成测试用的假登录页, 选一个身份继续 (用于验证多用户数据隔离)。</p>
      {buttons}
      <form action="" method="get">
        <input type="hidden" name="jumpto" value="{jumpto}"/>
        <input type="hidden" name="appid" value="{appid}"/>
        <input name="__pick" placeholder="或输入任意 LDAP, 如 carol"/>
        <button type="submit">登录</button>
      </form>
    </body></html>
    """
    return HTMLResponse(html)


# login 页提交后(带 __pick) 走中间件: 302 回 jumpto?code=code-<user>
@app.middleware("http")
async def _pick_redirect(request, call_next):
    if request.url.path == "/auth/sso/login" and "__pick" in request.query_params:
        user = (request.query_params.get("__pick") or "").strip().lower()
        jumpto = request.query_params.get("jumpto") or "/"
        sep = "&" if "?" in jumpto else "?"
        return RedirectResponse(url=f"{jumpto}{sep}code=code-{user}", status_code=302)
    return await call_next(request)


@app.get("/auth/sso/api/check_code")
def check_code(code: str = "", appid: str = "", appkey: str = "") -> JSONResponse:
    user = _user_from_code(code)
    if not user:
        return JSONResponse({"errno": 1, "errmsg": "bad code"})
    return JSONResponse({"errno": 0, "data": {"ticket": f"tkt-{user}", "username": user}})


@app.get("/auth/sso/api/check_ticket")
def check_ticket(ticket: str = "", appid: str = "") -> JSONResponse:
    user = _user_from_ticket(ticket)
    if not user:
        return JSONResponse({"errno": 1, "errmsg": "bad ticket"})
    return JSONResponse({"errno": 0, "data": {"username": user}})


@app.get("/auth/ldap/logout", response_class=HTMLResponse)
def logout() -> HTMLResponse:
    return HTMLResponse("<html><body><h3>已登出 (mock)</h3></body></html>")
