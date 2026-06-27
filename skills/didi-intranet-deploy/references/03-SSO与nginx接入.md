# 滴滴 SSO + nginx 前置鉴权

滴滴内部 SSO 是自研类 CAS(非 OAuth2/OIDC)。核心是 nginx 用 auth_request 在请求到
后端之前完成鉴权, 再用 HTTP 头把用户名透传给后端。直接用 assets/ 里的成熟模板。

## 架构

```
浏览器 -> 麒麟域名 router -> 容器:
   nginx :80 --auth_request--> sso-sidecar(校验 cookie 里的 ticket)
        |鉴权过, 注入 X-SSO-User 头
        v
   后端 :8501 (只听 127.0.0.1)
```

为什么要 sidecar: SSO 的 code->ticket 兑换、ticket 校验是 HTTP 调用 + 写加密 cookie,
不适合塞进 nginx; 也不适合塞进 Streamlit(拿不到 request)。独立一个小 FastAPI 最干净,
和 nginx/后端同容器跑(supervisord 管)。

## SSO 三接口 (默认 mis.diditaxi.com.cn)
- 未登录跳转: `302 /auth/sso/login?appid=<APP_ID>&jumpto=<回跳url>`
- code 换 ticket: `GET /auth/sso/api/check_code`(回调里用)
- 校验 ticket: `GET /auth/sso/api/check_ticket`(每次请求拦截器用)
- 统一登出: `GET /auth/ldap/logout`

## 登录流程
1. nginx 子请求 `/_sso_check` -> sidecar `/sso/check`: 读 cookie 里 ticket, 调 check_ticket
2. 无效 -> nginx error_page 401 -> 302 到 SSO 登录页(带 appid + jumpto=回调地址)
3. 登录后 SSO 把 code 拼到回调地址 -> sidecar `/sso/callback`: code 换 ticket, Fernet 加密写 cookie, 302 回业务页
4. 后续请求带 cookie, 重复 1, 通过则 nginx 注入 `X-SSO-User` 反代给后端

## 踩坑清单 (实战总结)
1. code 只能用一次、60s 过期。登录/回调/API 域名必须同环境, 否则无限 302。
2. nginx 反代 Streamlit 必须透传 WebSocket: `proxy_http_version 1.1` + Upgrade/Connection 头, 否则页面假死。
3. 回跳地址用 `$http_host`(含端口), 不要用 `$host`(丢非标端口)。生产 80/443 两者等价。
4. 本地走 http 时 cookie 的 Secure 要关(`SSO_COOKIE_SECURE=0`), 否则浏览器不存 cookie; 生产 https 必须为 1。
5. nginx 反代/网关读超时要调大(>5min), 适配长任务同步等待。
6. Streamlit 在反代下 XSRF/CORS 校验会误杀, 用 `--server.enableXsrfProtection false --server.enableCORS false`(鉴权由 nginx 保证)。

## SSO 返回字段不确定怎么办
不同环境 check_ticket/check_code 返回字段名可能不同。assets/deploy/sso_sidecar/main.py
已做宽容解析: 成功标志兼容 errno/errcode/code/ret/status; 用户名兼容
username/user/empId/email/loginName(email 自动取 @ 前缀); ticket 兼容 ticket/st/token;
支持嵌在 data/result 里。所以联调时大概率不用改代码。
万一仍取不到用户名: 拿到 check_ticket 真实返回 JSON, 按它调整 _USER_KEYS / _SUCCESS_KEYS 即可。

## 部署需要的环境变量
见 05-工单与上线.md 的环境变量清单(SSO_APP_ID/KEY/COOKIE_SECRET/LOGIN_BASE 等)。
