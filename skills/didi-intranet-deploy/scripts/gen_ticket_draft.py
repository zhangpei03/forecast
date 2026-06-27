#!/usr/bin/env python3
"""生成 BPM/SSO/域名/RDS 工单草稿 + OE 环境变量清单 + Fernet cookie 密钥。

把"每次都要查链接、拼回调地址、生成密钥"的重复工作一次产出。用法:

    python3 gen_ticket_draft.py \
        --app-name "Forecast Lab" \
        --domain forecast.xiaojukeji.com \
        --owner parkerzhang \
        --image hub.xiaojukeji.com/parkerzhang/forecast:v1 \
        --db forecast_lab \
        [--out 工单草稿.md]

不传 --out 时打印到 stdout。Fernet 密钥每次随机生成。
"""
from __future__ import annotations

import argparse


def _fernet_key() -> str:
    try:
        from cryptography.fernet import Fernet
        return Fernet.generate_key().decode()
    except Exception:
        import base64
        import os
        return base64.urlsafe_b64encode(os.urandom(32)).decode()


TPL = """# {app} 上线工单草稿(照抄即可)

> 镜像: {image}
> 域名: {domain} | 回调: https://{domain}/sso/callback | 负责人: {owner}

## 工单1 - SSO 接入(BPM)
打开: https://bpm.didichuxing.com/process/form/bykey/sso_upm_app_add_v3?tenantId=SSO

| 字段 | 值 |
|---|---|
| 子系统名称 | {app} |
| 主页地址 | https://{domain}/ |
| 回调地址 | https://{domain}/sso/callback |
| 管理员账号前缀 | {owner} |
| 系统所属环境 | 线上 |
| 是否开放权限申请 | 否 |

提交后等邮件下发 APP_ID / APP_KEY。

## 工单2 - 内网域名(BPM)
打开: https://bpm.didichuxing.com/process/form/bykey/information_security_domain_insert?tenantId=BPM

| 字段 | 值 |
|---|---|
| 申请域名 | {domain} |
| 业务线 | 麒麟 |
| 部署机房 | 与 OE 部署机房一致 |
| 解析目标 | OE 部署单元 IP(解析到麒麟 router) |

域名必须与工单1回调地址一致。

## 工单3 - 安全评估(SDL)
打开: https://sdl.xiaojukeji.com/sdl/dorado
- 系统名称: {app} | 访问域名: https://{domain}
- 鉴权: 公司 SSO(nginx auth_request 前置)

## 工单4 - MySQL RDS(数据库平台)
- 库名: {db} | 机房: 与 OE 同机房
- 拿到后记下 host / port / user / password

## OE 部署环境变量(把 <尖括号> 换成真实值)
```
DATABASE_URL=mysql+pymysql://{db}:<RDS密码>@<RDS_HOST>:3306/{db}?charset=utf8mb4
FORECAST_LAB_RUNTIME_DIR=/data
MAX_CONCURRENT_TRAINING=3
SSO_APP_ID=<工单1下发>
SSO_APP_KEY=<工单1下发>
SSO_COOKIE_SECRET={fernet}
SSO_LOGIN_BASE=http://mis.diditaxi.com.cn/auth/sso/login
CALLBACK_SCHEME=https
```
不要设置: SSO_DEV_FAKE_USER / *_DEV_USER / SSO_CHECK_CODE_URL / SSO_CHECK_TICKET_URL
(留空走默认 mis.diditaxi.com.cn)

> SSO_COOKIE_SECRET 是随机生成的 Fernet 密钥, 属机密, 勿发公开群。

## 联调验证
1. 访问 https://{domain}/ -> 应跳公司 SSO 登录
2. 登录后页面显示当前用户
3. 跑通核心流程
4. 拉 1~2 同事验证互相看不到对方数据 -> 开放给目标用户

若登录后取不到用户名: 把 SSO check_ticket 真实返回 JSON 拿来, 对齐 sidecar 字段解析。
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--app-name", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--owner", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--db", default="app_db")
    ap.add_argument("--out")
    a = ap.parse_args()
    text = TPL.format(app=a.app_name, domain=a.domain, owner=a.owner,
                      image=a.image, db=a.db, fernet=_fernet_key())
    if a.out:
        from pathlib import Path
        Path(a.out).write_text(text, encoding="utf-8")
        print(f"已写入 {a.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
