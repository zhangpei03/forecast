#!/usr/bin/env python3
"""项目内网部署体检 (阶段0)。

只读扫描一个本地项目, 输出"距离内网 SSO 部署还差什么"的结构化报告。
不修改任何文件。用法:

    python3 preflight_check.py <项目根目录>

检查维度: 技术栈/启动方式、鉴权现状、用户与数据隔离、存储后端、
打包制品、外网出口/重依赖、配置化程度。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def _read(p: Path, limit: int = 200_000) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")[:limit]
    except Exception:
        return ""


def _grep_files(root: Path, patterns: list[str], exts: tuple[str, ...]) -> list[str]:
    hits = []
    rx = re.compile("|".join(patterns), re.I)
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix not in exts:
            continue
        if any(seg in p.parts for seg in (".venv", "node_modules", ".git", "__pycache__")):
            continue
        if rx.search(_read(p, 50_000)):
            hits.append(str(p.relative_to(root)))
        if len(hits) >= 20:
            break
    return hits


def check(root: Path) -> dict:
    report: dict = {"root": str(root), "findings": {}, "gaps": [], "ok": []}
    f = report["findings"]

    def has(name: str) -> bool:
        return (root / name).exists()

    f["pyproject"] = has("pyproject.toml")
    f["requirements"] = has("requirements.txt")
    f["package_json"] = has("package.json")
    f["uv_lock"] = has("uv.lock")
    pyproj = _read(root / "pyproject.toml")
    f["streamlit"] = "streamlit" in pyproj or bool(_grep_files(root, [r"import streamlit"], (".py",)))
    f["fastapi"] = "fastapi" in pyproj.lower()
    f["flask"] = "flask" in pyproj.lower()
    f["django"] = "django" in pyproj.lower()

    auth_hits = _grep_files(root, [r"\bsso\b", r"auth_request", r"X-SSO-User", r"login_required",
                                   r"oauth", r"\bjwt\b"], (".py", ".js", ".ts", ".conf"))
    f["existing_auth_hits"] = auth_hits

    owner_hits = _grep_files(root, [r"owner_ldap", r"current_user", r"X-SSO-User"], (".py",))
    f["owner_isolation_hits"] = owner_hits

    f["sqlite"] = bool(_grep_files(root, [r"sqlite3\.connect", r"sqlite:///"], (".py", ".toml", ".env")))
    f["mysql"] = bool(_grep_files(root, [r"mysql\+pymysql", r"pymysql", r"mysqlclient"], (".py", ".toml")))
    f["postgres"] = bool(_grep_files(root, [r"psycopg", r"postgresql://"], (".py", ".toml")))

    f["dockerfile"] = has("Dockerfile")
    f["compose"] = has("docker-compose.yml") or has("compose.yaml")
    f["nginx_conf"] = bool(list(root.rglob("nginx*.conf")))

    f["nvidia_cuda_in_lock"] = "nvidia-" in _read(root / "uv.lock", 500_000)
    f["torch"] = "torch" in pyproj.lower() or "torch" in _read(root / "uv.lock", 500_000).lower()
    f["external_http"] = _grep_files(root, [r"requests\.(get|post)\(['\"]https?://(?!localhost)",
                                            r"httpx\.(get|post)\(['\"]https?://(?!localhost)"], (".py",))

    f["reads_env"] = bool(_grep_files(root, [r"os\.environ", r"getenv", r"load_dotenv"], (".py",)))
    f["env_example"] = has(".env.example")

    g = report["gaps"]
    ok = report["ok"]
    framework = next((n for n, v in [("streamlit", f["streamlit"]), ("fastapi", f["fastapi"]),
                                     ("flask", f["flask"]), ("django", f["django"])] if v), "未知")
    f["framework"] = framework

    if not auth_hits:
        g.append("无任何鉴权: 需接入 SSO(推荐 nginx auth_request 前置, 见 references/03)")
    else:
        ok.append(f"已发现鉴权相关代码: {auth_hits[:3]}")

    if not owner_hits:
        g.append("无用户/数据隔离: 需引入 owner_ldap 贯穿 DB+文件(见 references/02)")
    else:
        ok.append("已发现 owner/current_user 相关代码")

    if f["sqlite"] and not f["mysql"]:
        g.append("仅 SQLite: 多用户共享部署需换 MySQL(见 references/02)")
    if f["mysql"]:
        ok.append("已支持 MySQL")

    if not f["dockerfile"]:
        g.append("无 Dockerfile: 需容器化(见 references/04 + assets/ 模板)")
    else:
        ok.append("已有 Dockerfile")

    if framework == "streamlit":
        g.append("Streamlit 应用: 鉴权必须放 nginx 层(拿不到 request), 用户经 X-SSO-User 头透传")

    if f["nvidia_cuda_in_lock"]:
        g.append("依赖含 nvidia-*-cu12(GPU torch): 无 GPU 部署建议钉 CPU torch 瘦身(见 references/04)")

    if f["external_http"]:
        g.append(f"存在外网 HTTP 调用, 内网需走代理或关闭: {f['external_http'][:3]}")

    if not f["reads_env"]:
        g.append("配置疑似硬编码: 关键项(DB/SSO)应改为读环境变量")

    return report


def main() -> int:
    if len(sys.argv) < 2:
        print("用法: python3 preflight_check.py <项目根目录>", file=sys.stderr)
        return 2
    root = Path(sys.argv[1]).resolve()
    if not root.is_dir():
        print(f"目录不存在: {root}", file=sys.stderr)
        return 2
    report = check(root)

    print("=" * 60)
    print(f"内网部署体检: {root}")
    print(f"框架识别: {report['findings']['framework']}")
    print("=" * 60)
    print("\n[已具备]")
    for s in report["ok"]:
        print(f"  + {s}")
    print("\n[待补缺口]")
    for s in report["gaps"]:
        print(f"  - {s}")
    print("\n[原始 findings]")
    print(json.dumps(report["findings"], ensure_ascii=False, indent=2))
    print("\n下一步: 按 SKILL.md 主流程, 缺口逐项用 references/02~05 + assets/ 模板补齐。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
