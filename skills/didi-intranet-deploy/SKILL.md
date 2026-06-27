---
name: didi-intranet-deploy
description: 把一个本地运行的 Web 应用(Streamlit/FastAPI/Flask/Django 等)改造并部署到滴滴公司内网, 供多个员工通过公司 SSO 登录使用, 且实现用户级数据隔离。覆盖从项目体检、数据隔离改造、SSO+nginx 接入、容器化打包与构建修正(含 CPU torch 瘦身、内网镜像源、amd64), 到生成需要人工提交的 BPM/SSO/域名/SDL/RDS 工单草稿与 OE 部署环境变量。当用户想把本地项目/工具/Demo 部署到滴滴内网、让同事访问、接入公司 SSO 登录、做多用户数据隔离、申请内网域名(*.xiaojukeji.com)、走麒麟/OE 部署、或提到 mis.diditaxi.com.cn / auth_request / sso_upm_app_add_v3 时, 使用本技能。
---

# 滴滴内网部署 (本地应用 -> 内网 SSO 多用户)

把本地 Web 应用改造成"内网部署、SSO 登录、多用户数据隔离"的生产形态, 并产出人工提单所需信息。
适用于 Streamlit / FastAPI / Flask / Django 等 Python Web 应用(Streamlit 有特殊处理, 见下)。

## 五阶段主流程

按顺序推进。每阶段先读对应 reference, 再用 scripts/assets 落地。不要跳过阶段0。

### 阶段0 - 体检
先跑体检脚本, 看清"还差什么", 再动手:
```
python3 scripts/preflight_check.py <项目根目录>
```
然后读 references/01-检查清单.md 人工确认七项(框架/鉴权/隔离/存储/打包/外网依赖/配置化),
并和用户对齐"改造 vs 重写"(默认改造, 不轻易重写前端)。

### 阶段1 - 数据隔离改造 (多用户共享的核心)
读 references/02-数据隔离改造.md。把 owner_ldap 贯穿三处:
数据库(SQL 强制 WHERE owner)、文件(按用户分桶目录)、后台子进程(命令行透传)。
当前用户来自 X-SSO-User 头, 复用 assets/auth.py.template。
SQLite 多用户场景换 MySQL(SQLAlchemy + database_url)。务必加隔离单测。

### 阶段2 - SSO + nginx 接入
读 references/03-SSO与nginx接入.md。鉴权放 nginx auth_request 前置层,
用 sidecar 处理 SSO 三接口, 用 X-SSO-User 头把身份透传给后端。
直接复制 assets/deploy/(nginx.conf / sso_sidecar/main.py / supervisord.conf / entrypoint.sh)。
Streamlit 必须走这条路(它拿不到 request)。sidecar 已做 SSO 返回字段宽容解析, 大概率不用改。

### 阶段3 - 容器化打包
读 references/04-打包与构建修正.md。用 assets/Dockerfile.template 单容器跑
nginx+sidecar+后端。重点修正: 内网基础镜像(build-arg)、CPU torch 瘦身(避开 nvidia-cu12)、
内网 PyPI 源、--platform linux/amd64、模型权重预下载。
强烈建议先用 docker-compose 起 后端+MySQL+mock-sso 本地验证整条链路(含隔离), 再上线。
构建+推送:
```
docker build --platform linux/amd64 -t hub.xiaojukeji.com/<user>/<app>:v1 .
docker push hub.xiaojukeji.com/<user>/<app>:v1
```

### 阶段4 - 工单与上线 (人工提单)
读 references/05-工单与上线.md。用脚本一键生成填好值的工单草稿 + 环境变量 + 密钥:
```
python3 scripts/gen_ticket_draft.py --app-name "<应用名>" --domain <域名> \
    --owner <ldap> --image hub.xiaojukeji.com/<user>/<app>:v1 --db <库名> --out 工单草稿.md
```
产出 4 个工单(SSO/域名/SDL/RDS)+ OE 环境变量 + Fernet 密钥。
这些工单只能由用户在公司系统提交; 把草稿交给用户, 并提示密钥保密。

## 关键原则 (踩坑沉淀)

- 隔离的最后防线在数据访问层(SQL 强制 owner), 不靠上层自觉。
- Streamlit 不能自己接 SSO -> 一律 nginx auth_request 前置 + X-SSO-User 头透传。
- ML 依赖钉 CPU torch, 否则 nvidia-cu12 几 GB 拖垮内网构建。
- 本地 arm64 不能推 amd64 OE -> 构建必带 --platform linux/amd64。
- 改动必须不破坏本地开发(DB/用户/路径都要有本地降级)。
- nginx 回跳用 $http_host; 本地 http 关 cookie Secure; WebSocket 必须透传。

## 资源索引

- references/01-检查清单.md - 阶段0 体检七项 + 改造vs重写判断
- references/02-数据隔离改造.md - owner 贯穿 DB/文件/子进程 + MySQL 切换 + 隔离单测
- references/03-SSO与nginx接入.md - SSO 三接口、auth_request 流程、字段宽容解析、踩坑
- references/04-打包与构建修正.md - Dockerfile/CPU torch/内网源/amd64/构建排障
- references/05-工单与上线.md - BPM/SSO/域名/SDL/RDS 工单 + OE 环境变量 + 灰度
- scripts/preflight_check.py - 只读体检, 输出缺口清单(阶段0)
- scripts/gen_ticket_draft.py - 生成工单草稿+环境变量+Fernet密钥(阶段4)
- assets/Dockerfile.template、assets/deploy/*、assets/auth.py.template - 可直接复用的生产模板
