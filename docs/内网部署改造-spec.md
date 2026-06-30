# Forecast Lab 内网部署改造 Spec

> **版本**: v1.0 · 2026-06-25
> **作者**: parkerzhang
> **状态**: 待评审

## 0. 背景与目标

### 0.1 现状（基于代码探查）
- 单机 Streamlit 应用，Python 3.12 + uv，UI/逻辑/存储一体
- 默认 `localhost:8501`，**无任何鉴权**，**无 user 概念**
- 数据全落本地 `runtime/`：SQLite (`app.db`) + 上传文件 + 实验产物
- 训练任务由 `subprocess.Popen` 拉子进程跑（约 3 分钟一轮）
- 无 Dockerfile、无部署制品、无环境配置化

### 0.2 目标
- 部署到滴滴内网，10~50 人通过 `forecast.xiaojukeji.com` 访问
- 接入公司 SSO，所有访问者必须登录
- **用户级数据隔离**：每个用户只能看到/管理自己的实验和数据
- DB 切换为公司 MySQL（RDS），文件落容器持久卷
- 保留 Streamlit 不动（不重写前端）

### 0.3 不在本期范围
- React + Antd 前端重写（待 P0 上线 1 个月后根据反馈决定）
- 多副本部署 / 高可用 / 灰度
- 异步任务队列（Celery/Redis）—— 训练 3 分钟内同步等待可接受
- 权限分级（管理员/普通用户）—— 全员平权，仅按 owner 隔离
- 对象存储（GIFT）—— 用持久卷替代

---

## 1. 架构

```
员工浏览器
   │   https://forecast.xiaojukeji.com
   ▼
麒麟 Router (内网域名解析)
   │
   ▼
OE 部署容器 (单副本)
┌────────────────────────────────────────────────────┐
│  nginx :80                                          │
│    ├─ location /sso/*        → 反代 sso-sidecar    │
│    ├─ auth_request /sso/check  (拦所有业务请求)    │
│    └─ location /             → 反代 streamlit:8501 │
│         + proxy_set_header X-SSO-User $sso_user    │
│                                                     │
│  sso-sidecar (FastAPI :9000, 仅 nginx 内部访问)    │
│    ├─ GET /sso/check         (cookie→ticket 校验)  │
│    ├─ GET /sso/callback      (code→ticket 写cookie)│
│    └─ GET /sso/logout                              │
│                                                     │
│  streamlit :8501 (127.0.0.1)                       │
│    └─ 从 st.context.headers["X-SSO-User"] 取身份   │
└────────────────────────────────────────────────────┘
   │
   ├─ MySQL (公司 RDS)              # experiments / runs 两表
   └─ /data 持久卷                   # uploads/<ldap>/, experiments/<ldap>/<exp_id>/
```

**关键决策**：
- SSO 鉴权在 **nginx 层**完成，Streamlit 不引入任何鉴权代码（Streamlit 拿不到 request，无法插中间件）
- 用户身份通过 **HTTP header `X-SSO-User`** 透传给 Streamlit，Streamlit 通过 `st.context.headers` 读取（≥1.37 支持，当前 1.58 OK）
- nginx + sso-sidecar + streamlit **跑在同一容器**，用 supervisord 管理（部署简单，无需引入 sidecar pattern）

---

## 2. 改造清单（按文件级）

> 命名约定：`[NEW]` 新增文件 · `[MOD]` 修改文件 · `[CFG]` 配置变更

### 2.1 鉴权与用户身份（核心新增）

#### `[NEW] src/core/auth.py`
提供"当前用户"上下文与降级策略：

```python
# 伪代码
def get_current_user() -> str:
    """从 Streamlit 请求头取 LDAP；本地开发回退到 env 或 'local-dev'。"""
    try:
        ldap = st.context.headers.get("X-SSO-User")
        if ldap:
            return ldap.strip().lower()
    except Exception:
        pass
    # 本地开发降级
    return os.getenv("FORECAST_LAB_DEV_USER", "local-dev")

def require_user() -> str:
    """业务代码统一入口；缺失 user 时显式报错（生产应不会发生，nginx 已拦截）。"""
    user = get_current_user()
    if not user:
        st.error("未识别到登录用户，请刷新页面重新登录"); st.stop()
    return user
```

**约束**：
- 所有 page / repository 调用一律走 `require_user()`，**不准在业务代码里 grep header**
- LDAP 一律小写存储，避免大小写歧义

#### `[NEW] deploy/sso_sidecar/main.py`（FastAPI）
三个路由对应 `mis.diditaxi.com.cn` 三接口：

| 路由 | 行为 |
|---|---|
| `GET /sso/check` | 取 cookie 里的 `sso_ticket` → 调 `check_ticket`；200 返回 + `X-SSO-User` header / 401 触发 nginx 重定向 |
| `GET /sso/callback?code=xxx&jumpto=xxx` | 调 `check_code` 换 ticket → 加密写 cookie → 302 回 `jumpto` |
| `GET /sso/logout` | 清 cookie → 302 到 `mis.diditaxi.com.cn/auth/ldap/logout` |

实现细节：
- ticket 加密用 `cryptography.Fernet`，密钥从 env `SSO_COOKIE_SECRET` 读
- cookie 属性：`HttpOnly; Secure; SameSite=Lax; Path=/; Max-Age=28800`（8h）
- 60s code 防重放：调过 `check_code` 后立即丢弃，依靠 SSO 服务端单次性
- `APP_ID` / `APP_KEY` 从 env 读，**不入仓**

#### `[NEW] deploy/nginx.conf`
核心配置（节选）：

```nginx
server {
  listen 80;
  server_name forecast.xiaojukeji.com;
  client_max_body_size 60M;     # 配合 MAX_UPLOAD_MB=50 + 余量
  proxy_read_timeout 600s;       # 训练同步等待 + 余量（>5min）

  # SSO 三接口直通 sidecar
  location /sso/ { proxy_pass http://127.0.0.1:9000; }

  # 业务请求统一走 auth_request
  location / {
    auth_request /_sso_check;
    auth_request_set $sso_user $upstream_http_x_sso_user;

    # 鉴权失败 → 302 到 SSO 登录页
    error_page 401 = @sso_login;

    proxy_pass http://127.0.0.1:8501;
    proxy_set_header X-SSO-User $sso_user;
    proxy_set_header Host $host;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;       # Streamlit WebSocket 必须
    proxy_set_header Connection "upgrade";
  }

  location = /_sso_check {
    internal;
    proxy_pass http://127.0.0.1:9000/sso/check;
    proxy_pass_request_body off;
    proxy_set_header Content-Length "";
    proxy_set_header X-Original-URI $request_uri;
  }

  location @sso_login {
    return 302 http://mis.diditaxi.com.cn/auth/sso/login?appid=$SSO_APP_ID&jumpto=https://$host$request_uri;
  }
}
```

### 2.2 数据隔离（user_ldap 字段贯穿）

#### `[MOD] src/storage/sqlite.py` → 改名为 `src/storage/db.py`

换成 **SQLAlchemy 2.0 + 连接池**，同时保留 schema 初始化：

- 新增 `engine` 单例（连接池 size=10，pre_ping 防断连）
- 通过 `FORECAST_LAB_DB_URL` env 切换：
  - 本地：`sqlite:///runtime/app.db`
  - 生产：`mysql+pymysql://user:pass@rds-host:3306/forecast_lab?charset=utf8mb4`
- DDL 改造（用 Alembic 迁移而非裸 SQL）：

```sql
CREATE TABLE experiments (
  id              VARCHAR(64)  PRIMARY KEY,
  owner_ldap      VARCHAR(64)  NOT NULL,        -- ★ 新增
  name            VARCHAR(255) NOT NULL,
  status          VARCHAR(32)  NOT NULL,
  ... (原字段全部保留，TEXT→LONGTEXT，REAL→DOUBLE) ...
  created_at      DATETIME     NOT NULL,
  updated_at      DATETIME     NOT NULL,
  KEY idx_owner_created (owner_ldap, created_at DESC),
  KEY idx_owner_status  (owner_ldap, status)
);

CREATE TABLE runs (
  id              VARCHAR(64)  PRIMARY KEY,
  experiment_id   VARCHAR(64)  NOT NULL,
  owner_ldap      VARCHAR(64)  NOT NULL,        -- ★ 新增（冗余但查询友好）
  ...
  KEY idx_exp (experiment_id),
  KEY idx_owner_status (owner_ldap, status)
);
```

#### `[NEW] alembic/` 目录
- `alembic.ini` + `env.py`
- `versions/0001_init.py` — 建表（与上面 DDL 一致）
- `versions/0002_add_owner_ldap.py` — 从 SQLite 迁 MySQL 时补字段（如果有存量数据）
- 启动时不自动迁移，由 `make db-upgrade` 显式触发，避免多副本同时迁移

#### `[MOD] src/repositories/experiment_repository.py`

**所有方法签名加 `owner_ldap` 强制参数**，并在 SQL 层强制 WHERE：

| 方法 | 改造 |
|---|---|
| `list_experiments(self, owner_ldap)` | `WHERE owner_ldap = :owner ORDER BY created_at DESC` |
| `get_experiment(self, experiment_id, owner_ldap)` | `WHERE id=:id AND owner_ldap=:owner`，越权访问返回 None |
| `create_experiment(self, config, *, owner_ldap, ...)` | INSERT 时写入 owner_ldap |
| `update_status / update_results` | UPDATE 加 `WHERE id=:id AND owner_ldap=:owner` |
| `create_run(self, experiment_id, *, owner_ldap, log_path)` | 同上 |
| `update_run(self, run_id, *, owner_ldap=None, ...)` | 子进程更新 run 时 owner 校验可选（已有 run_id 即可信） |
| `latest_run(self, experiment_id, owner_ldap)` | JOIN/WHERE 双校验 |

**安全原则**：repository 层是数据隔离最后一道防线，**任何越权请求都必须返回空或抛 PermissionError**，不能依赖 page 层自觉。

#### `[MOD] src/domain/models.py`
`ExperimentConfig` 加字段：
```python
owner_ldap: str   # 必填，由 page 层从 require_user() 注入
```

#### `[MOD] pages/experiments.py`、`pages/create_experiment.py`、`pages/run_status.py`、`pages/result_analysis.py`

每个 page 顶部统一加：
```python
from src.core.auth import require_user
current_user = require_user()
```

所有 `repository.xxx(...)` 调用都补上 `owner_ldap=current_user`。

### 2.3 文件存储隔离

#### `[MOD] src/core/config.py`

从单一 `runtime_dir` 派生改为支持 user 维度：

```python
@dataclass(frozen=True)
class AppSettings:
    runtime_dir: Path = Path(os.getenv("FORECAST_LAB_RUNTIME_DIR", "runtime"))
    db_url: str       = os.getenv("FORECAST_LAB_DB_URL", f"sqlite:///{runtime_dir}/app.db")

    def upload_dir(self, owner_ldap: str) -> Path:
        return self.runtime_dir / "uploads" / owner_ldap

    def experiments_dir(self, owner_ldap: str) -> Path:
        return self.runtime_dir / "experiments" / owner_ldap
```

#### `[MOD] src/storage/file_store.py`

`get_experiment_dir` 签名加 `owner_ldap`：
```python
def get_experiment_dir(settings, owner_ldap: str, experiment_id: str) -> Path:
    path = settings.experiments_dir(owner_ldap) / experiment_id
    ...
```

`ensure_runtime_dirs` 改为只创建顶层 `uploads/`、`experiments/`；用户子目录由首次访问时按需创建（避免新用户登录时全量遍历）。

### 2.4 训练子进程身份透传

#### `[MOD] src/jobs/job_runner.py`
```python
def start_training_job(*, settings, repository, experiment_id, owner_ldap: str) -> int:
    experiment_dir = get_experiment_dir(settings, owner_ldap, experiment_id)
    ...
    process = subprocess.Popen([
        sys.executable, "-m", "src.jobs.train_worker",
        "--experiment-id", experiment_id,
        "--run-id", run_id,
        "--owner-ldap", owner_ldap,        # ★ 新增
        "--db-url", settings.db_url,       # ★ 改为 db_url
        "--runtime-dir", str(settings.runtime_dir),
    ], ...)
```

#### `[MOD] src/jobs/train_worker.py`
- `_parse_args()` 增加 `--owner-ldap` / 替换 `--database-path` 为 `--db-url`
- `main()` 中所有路径拼接、repository 调用都使用 `args.owner_ldap`
- 子进程 import 路径不变（容器内 PYTHONPATH 已通过 uv 注入）

### 2.5 容器化

#### `[NEW] Dockerfile`
多阶段构建，公司内网基础镜像：

```dockerfile
# Stage 1: builder
FROM hub.xiaojukeji.com/base/python:3.12-slim-bookworm AS builder
ENV UV_INDEX_URL=https://pypi.intra.xiaojukeji.com/simple
ENV UV_LINK_MODE=copy
RUN pip install --no-cache-dir uv==0.5.*
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Stage 2: runtime
FROM hub.xiaojukeji.com/base/python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx supervisor && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
COPY . /app
WORKDIR /app

# nginx + supervisor 配置
COPY deploy/nginx.conf /etc/nginx/sites-enabled/default
COPY deploy/supervisord.conf /etc/supervisor/supervisord.conf

# 预下载 AutoGluon 权重（关键！避免线上首跑外网）
RUN python scripts/prefetch_models.py

EXPOSE 80
VOLUME /data
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/supervisord.conf"]
```

#### `[NEW] deploy/supervisord.conf`
```ini
[supervisord]
nodaemon=true

[program:nginx]
command=/usr/sbin/nginx -g 'daemon off;'
autorestart=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0

[program:sso-sidecar]
command=uvicorn deploy.sso_sidecar.main:app --host 127.0.0.1 --port 9000
autorestart=true

[program:streamlit]
command=streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --server.enableXsrfProtection false --server.enableCORS false
environment=FORECAST_LAB_RUNTIME_DIR="/data",FORECAST_LAB_DB_URL="%(ENV_FORECAST_LAB_DB_URL)s"
autorestart=true
```

> ⚠️ XSRF/CORS 关掉的原因：Streamlit 在 nginx 反代下，Origin/Host 校验会误杀。鉴权由 nginx 层保证。

#### `[NEW] scripts/prefetch_models.py`
在构建时下载 AutoGluon TimeSeries 用到的深度模型权重，避免生产首跑联网拉权重（公司内网可能拉不到）。

### 2.6 环境与配置

#### `[MOD] .env.example`（补全）
```bash
# 运行时数据目录
FORECAST_LAB_RUNTIME_DIR=/data

# 数据库
FORECAST_LAB_DB_URL=mysql+pymysql://forecast:CHANGE_ME@rds-xxx.intra.xiaojukeji.com:3306/forecast_lab?charset=utf8mb4

# SSO（向 SSTG 申请后填入）
SSO_APP_ID=changeme
SSO_APP_KEY=changeme
SSO_COOKIE_SECRET=base64-fernet-key
SSO_LOGIN_URL=http://mis.diditaxi.com.cn/auth/sso/login
SSO_CHECK_CODE_URL=http://mis.diditaxi.com.cn/auth/sso/api/check_code
SSO_CHECK_TICKET_URL=http://mis.diditaxi.com.cn/auth/sso/api/check_ticket
SSO_LOGOUT_URL=http://mis.diditaxi.com.cn/auth/ldap/logout

# 本地开发降级身份（生产不设置）
# FORECAST_LAB_DEV_USER=local-dev
```

#### `[MOD] pyproject.toml` 新增依赖
```toml
sqlalchemy>=2.0,<3
pymysql>=1.1,<2
alembic>=1.13,<2
fastapi>=0.115,<1
uvicorn>=0.30,<1
cryptography>=43,<46
httpx>=0.27,<1                # SSO sidecar 调 mis.diditaxi.com.cn
```

#### `[MOD] Makefile` 新增 target
```makefile
db-upgrade:    uv run alembic upgrade head
db-downgrade:  uv run alembic downgrade -1
sidecar-dev:   uv run uvicorn deploy.sso_sidecar.main:app --reload --port 9000
docker-build:  docker build --platform linux/amd64 -t hub.xiaojukeji.com/parkerzhang/forecast:v1 .
docker-push:   docker push hub.xiaojukeji.com/parkerzhang/forecast:v1
```

---

## 3. 数据迁移

### 3.1 是否需要迁移历史 SQLite 数据
**默认不迁**。原因：
- 现有 SQLite 是本地开发数据，无 owner 概念，强行赋一个 owner 没意义
- 上线前用户应重新创建实验
- 如果某些 demo 实验确实要保留，写一个 `scripts/migrate_sqlite_to_mysql.py`，由人工指定 `--owner-ldap parkerzhang` 赋归属

### 3.2 文件目录变更
- 旧：`runtime/uploads/<file>`、`runtime/experiments/<exp_id>/`
- 新：`/data/uploads/<ldap>/<file>`、`/data/experiments/<ldap>/<exp_id>/`
- 本地开发兼容：`FORECAST_LAB_DEV_USER=local-dev` 时落到 `runtime/uploads/local-dev/` 下，不污染老路径

---

## 4. 申请流程（并行启动）

| 项 | 平台 | 链接 | 负责人 | 周期 |
|---|---|---|---|---|
| SSO 申请 | BPM | https://bpm.didichuxing.com/process/form/bykey/sso_upm_app_add_v3?tenantId=SSO | parkerzhang | 1~3d |
| 内网域名 `forecast.xiaojukeji.com` | BPM | https://bpm.didichuxing.com/process/form/bykey/information_security_domain_insert?tenantId=BPM | parkerzhang | 1~3d |
| 安全评估 | SDL | https://sdl.xiaojukeji.com/sdl/dorado | parkerzhang | 与域名并行 |
| MySQL RDS | 数据库平台 | (待补) | parkerzhang | 1~3d |
| OE 部署单元 | OE | (待补) | parkerzhang | 1d |
| 镜像仓库 | hub.xiaojukeji.com | 已就绪 (复用 parkerzhang namespace) | parkerzhang | - |

**SSO 工单填写要点**：
- 子系统名称：`Forecast Lab — 网约车财务预测评测台`
- 主页地址：`https://forecast.xiaojukeji.com/`
- 回调地址：`https://forecast.xiaojukeji.com/sso/callback`
- 管理员账号前缀：`parkerzhang`（暂时只放自己，后续可加）
- 环境：线上
- 是否开放权限申请：否（员工凭 LDAP 直接登录即可）

**域名工单填写要点**：
- 业务线：**麒麟**
- 部署机房：与 OE 部署机房一致
- IP/端口：解析到麒麟 router

---

## 5. 安全与合规

| 项 | 措施 |
|---|---|
| 鉴权 | nginx auth_request 拦截 100% 业务请求，未登录强制 302 |
| 数据隔离 | repository 层 SQL WHERE 强制 owner_ldap，文件路径按 LDAP 分桶 |
| Cookie 安全 | `HttpOnly` + `Secure` + `SameSite=Lax`，ticket 用 Fernet 加密 |
| 密钥管理 | `SSO_APP_KEY` / `SSO_COOKIE_SECRET` / DB 密码全部走 env，不入仓 |
| 越权防御 | 即使 page 层忘记带 owner，repository 层强制参数报错 |
| 上传限制 | nginx `client_max_body_size 60M` + Streamlit `maxUploadSize=50M` 双层 |
| 日志脱敏 | `run.log` 不打印用户上传的业务数据样本，仅打印列名与统计量 |
| 外网出口 | 容器内不直连外网；AutoGluon 模型权重在构建时预下载烤进镜像 |

---

## 6. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| Streamlit `st.context.headers` 不稳定 | 低 | 高 | 当前 Streamlit 1.58 已稳定支持；上线前在 staging 验证；保留 `FORECAST_LAB_DEV_USER` 降级 |
| nginx WebSocket 反代配置错 → Streamlit 假死 | 中 | 高 | `proxy_http_version 1.1` + `Upgrade/Connection` header 必须；上线前完整跑通一次创建实验流程 |
| SQLite→MySQL SQL 兼容性 | 中 | 中 | 全面引入 SQLAlchemy 2.0，所有 SQL 走 ORM/text() 参数化；提前在本地用 MySQL 容器联调 |
| AutoGluon 首次拉权重在内网失败 | 高 | 高 | 构建期 `scripts/prefetch_models.py` 烤进镜像；启动期再 fallback 一次本地 cache |
| 训练子进程超过 nginx 超时 | 低 | 中 | `proxy_read_timeout 600s`（10min 余量）；训练若超 5min 主动报警 |
| 50 人并发对单容器压力 | 低 | 中 | 训练同步等待 + 单容器，理论上同时跑训练数 ≤ CPU 核数；先按 4C8G 起，按需扩容 |
| 多人共享 `/data` 持久卷 IO 竞争 | 低 | 低 | 文件按用户隔离子目录；SQLite 已弃用 |
| 数据隔离漏洞（page 层忘传 owner） | 中 | 极高 | repository 层强制 `owner_ldap` 必填参数（Python type hint + 运行时校验）；写 pytest 覆盖每个 repository 方法 |

---

## 7. 测试

### 7.1 单元测试（必加）
`tests/unit/test_repository_isolation.py`：
- `test_user_a_cannot_see_user_b_experiments`
- `test_user_a_cannot_get_user_b_experiment_by_id`
- `test_user_a_cannot_update_user_b_experiment`
- `test_owner_ldap_normalized_to_lowercase`

`tests/unit/test_auth.py`：
- `test_require_user_from_header`
- `test_require_user_fallback_to_env_in_dev`
- `test_require_user_stops_when_missing`

### 7.2 集成测试（手动 checklist）
- [ ] 本地起 docker-compose（streamlit + nginx + sso-mock + mysql），完整跑通创建实验 → 训练 → 查看结果
- [ ] 用两个不同 mock user 登录，验证看不到对方数据
- [ ] 第二个用户尝试用第一个用户的 experiment_id 访问 URL，应 404 或空
- [ ] 文件上传/下载路径确实落在用户子目录

### 7.3 生产灰度
- 第 1 天：仅 parkerzhang 自己访问，跑 2~3 个实验
- 第 2~3 天：拉 2~3 个同事 LDAP 加进 admin 名单试用
- 第 4 天起：开放给目标 10~50 人

---

## 8. 实施计划

| 阶段 | 工作 | 工时 | 依赖 |
|---|---|---|---|
| **W1-D1~D2** | SQLAlchemy 引入 + repository 加 owner_ldap + 单测 | 2d | - |
| **W1-D3** | `src/core/auth.py` + page 层接入 + 文件路径隔离 | 1d | repository 完成 |
| **W1-D4~D5** | sso-sidecar + nginx 配置 + 本地 mock 联调 | 2d | - |
| **W2-D1** | Dockerfile + supervisord + 镜像构建推送 | 1d | sidecar 完成 |
| **W2-D2** | 启动 BPM × 2 + SDL + RDS 工单（并行等待） | 0.5d | - |
| **W2-D3~D4** | Alembic 迁移脚本 + 本地 MySQL 容器联调 | 2d | docker 完成 |
| **W2-D5** | OE 部署 + 拿到真 SSO `app_id/app_key` 联调 | 1d | 工单批复 |
| **W3-D1~D2** | 灰度 + bug fix + 性能验证 | 2d | - |
| **W3-D3** | 正式开放 | 0.5d | - |
| **合计** | | **~12 工作日（含 1.5d 缓冲）** | |

---

## 9. 验收标准

- [ ] 任何未登录请求访问 `https://forecast.xiaojukeji.com/` 都被 302 到 SSO 登录页
- [ ] 登录后 Streamlit 顶部能正确显示 `当前用户：xxxx@didiglobal.com`
- [ ] 用户 A 看不到用户 B 的任何实验、运行记录、上传文件
- [ ] 直接拼 URL 访问别人的 experiment_id 返回空/404
- [ ] 训练任务正常跑通，3 分钟内出结果
- [ ] 容器重启后所有用户的实验/上传文件仍在
- [ ] 镜像构建走公司 PyPI 镜像，不依赖外网
- [ ] 单测全通过，repository 越权防御 case 100% 覆盖

---

## 10. 后续演进（不在本期）

| 项 | 触发条件 |
|---|---|
| 多副本部署 | 同时在线 > 30，单容器 CPU > 70% |
| 异步任务队列（Celery + Redis） | 训练任务超 5min，或用户抱怨等待 |
| 对象存储（GIFT）替代持久卷 | 数据量 > 100GB 或需要跨副本共享 |
| React + Antd 前端重写 | 收到 ≥ 5 个明确的"Streamlit 体验不够"反馈 |
| 团队空间 / 共享实验 | 用户主动要求"我能不能看到组里其他人的实验" |
| 权限分级（admin/普通用户） | 出现需要管理员能力的场景（强制下线、清理数据等） |

---

## 附录 A：本地开发不破坏

所有改造对本地开发的影响：
- `make run` 仍可用，自动以 `local-dev` 身份运行
- `FORECAST_LAB_DB_URL` 未设置时回落到 `sqlite:///runtime/app.db`，无需 MySQL
- 文件路径变成 `runtime/uploads/local-dev/...` 和 `runtime/experiments/local-dev/<exp_id>/`
- 老的 `runtime/uploads/*` 和 `runtime/experiments/<exp_id>/` 数据不会被自动迁移，可手动 `mv` 到 `local-dev/` 子目录或 `make clean-runtime` 重置

## 附录 B：相关文档

- SSO 接入清单：已上传到 D-Chat 个人空间 `sso-接入清单.md`
- 麒麟平台：https://kylin.intra.xiaojukeji.com/
- 项目代码：git@git.xiaojukeji.com:report_forecast/report_forecast.git
