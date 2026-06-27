# 本地集成验证 (docker-compose)

一键起整条内网部署链路, 不接真 SSO, 用 mock 验证「登录拦截 → 用户透传 → 数据隔离」。

## 组件

| 服务 | 作用 | 端口(宿主) |
|---|---|---|
| `app` | 主镜像(nginx + sso-sidecar + streamlit) | http://localhost:8080 |
| `mock-sso` | 假 SSO(可选不同身份, 验证多用户隔离) | http://localhost:18080 |
| `mysql` | MySQL 8, 库 `forecast_lab` | localhost:13306 |

## 启动

```bash
docker compose up --build
```

> 需要本机 Docker daemon 在运行(Docker Desktop / colima 等)。

## 验证步骤

1. 浏览器打开 http://localhost:8080
   - 未登录会被 nginx 拦截 → 302 跳到 mock 登录页(localhost:18080)。
2. 选 **alice** 登录 → 回到应用, 左侧栏显示「当前用户：alice」。
3. 用 alice 创建一个实验(可用 `make demo-data` 生成的样例, 或 `sample_data/` 里的 xlsx)。
4. 退出: 访问 http://localhost:8080/sso/logout, 再打开首页, 这次选 **bob** 登录。
5. **隔离验证**: bob 的实验列表应为空, 看不到 alice 刚建的实验。
6. **越权验证**: 把 alice 实验详情页 URL 里的 `experiment_id` 复制给 bob 访问,
   应提示「实验不存在」(repository 层按 owner 过滤返回 None)。
7. 文件隔离: 进 `app` 容器看目录, 应是按用户分桶:
   ```bash
   docker compose exec app ls -R /data/experiments
   # /data/experiments/alice/<exp_id>/...
   # /data/experiments/bob/<exp_id>/...
   ```

## 链路要点(便于排障)

- **双地址**: nginx 302 与 cookie 回跳用浏览器可达的 `localhost:18080`;
  sidecar 调 `check_code/check_ticket` 用容器网络 `mock-sso:8080`。
- **cookie**: 本地走 http, 故 `SSO_COOKIE_SECURE=0`(生产 https 必须为 1)。
- **身份编码**: mock 的 ticket = `tkt-<user>`, code = `code-<user>`,
  不同用户拿到不同 ticket, sidecar 据此识别不同 LDAP。
- **不要设** `SSO_DEV_FAKE_USER`: 一旦设置所有人变同一身份, 无法验证隔离。

## 清理

```bash
docker compose down -v    # -v 连数据卷一起删
```

## 实测记录 (2026-06-27)

整条链路已在本地 docker compose 实测通过:

- 镜像构建: torch 钉 CPU wheel(`[tool.uv.sources]` + `torch` 入主依赖),
  避开 GPU 版 torch + nvidia-*-cu12, `uv sync --frozen` 约 2 分钟装 149 包
- 三进程: nginx / sso-sidecar / streamlit 经 supervisord 全部 RUNNING
- init_db 在真实 MySQL(forecast_lab) 建表, owner_ldap 列正确落库
- 鉴权拦截: 未登录访问 302 跳登录; /sso/check 无 cookie 返回 401
- 登录链路: code-alice -> check_code -> 加密 cookie -> 带 cookie /sso/check
  返回 200 + X-SSO-User: alice
- 数据隔离: alice 只见 exp_alice, bob 只见 exp_bob;
  alice 用 bob 的 experiment_id 越权 get 返回 None
- MySQL 底表确认 experiments.owner_ldap 按用户分

### 踩坑修复

- nginx @sso_login 的 jumpto 原用 $host(不含端口), 非标准端口下回跳会丢端口;
  已改为 $http_host(含端口), 生产 80/443 默认端口下两者等价, 不受影响。

### 常用排障命令

    docker compose logs app --tail 50
    docker compose exec mysql sh -c 'mysql -uforecast -pforecastpw forecast_lab -e "SELECT id,owner_ldap,status FROM experiments;"'
    docker compose exec app ls -R /data/experiments
    curl -s -o /dev/null -w "%{http_code} %{redirect_url}\n" http://localhost:8080/
