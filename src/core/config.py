from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from src.core.constants import RUNTIME_DIR


def _default_database_url() -> str:
    """优先读 DATABASE_URL 环境变量, 否则回退到本地 sqlite 文件(开发兜底)。"""
    url = os.environ.get("DATABASE_URL", "").strip()
    if url:
        return url
    sqlite_path = Path(os.environ.get("FORECAST_LAB_RUNTIME_DIR", RUNTIME_DIR)) / "app.db"
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{sqlite_path}"


def _default_max_concurrent() -> int:
    try:
        return max(1, int(os.environ.get("MAX_CONCURRENT_TRAINING", "3")))
    except ValueError:
        return 3


@dataclass(frozen=True)
class AppSettings:
    runtime_dir: Path = Path(os.environ.get("FORECAST_LAB_RUNTIME_DIR", RUNTIME_DIR))
    database_url: str = field(default_factory=_default_database_url)
    max_concurrent_training: int = field(default_factory=_default_max_concurrent)

    @property
    def database_path(self) -> Path:
        """向后兼容: 旧代码引用 .database_path 时仍可读到 sqlite 文件路径(若使用 sqlite URL)。"""
        return self.runtime_dir / "app.db"

    @property
    def upload_root(self) -> Path:
        """所有用户上传文件的根目录(其下按 owner_ldap 分桶)。"""
        return self.runtime_dir / "uploads"

    @property
    def experiments_root(self) -> Path:
        """所有用户实验产物的根目录(其下按 owner_ldap 分桶)。"""
        return self.runtime_dir / "experiments"

    def upload_dir(self, owner_ldap: str) -> Path:
        """某个用户的上传目录: runtime/uploads/<owner_ldap>/"""
        return self.upload_root / _safe_segment(owner_ldap)

    def experiments_dir(self, owner_ldap: str) -> Path:
        """某个用户的实验产物目录: runtime/experiments/<owner_ldap>/"""
        return self.experiments_root / _safe_segment(owner_ldap)


def _safe_segment(value: str) -> str:
    """把 LDAP 规整成安全的目录名(小写, 仅保留字母数字下划线中划线点)。"""
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in (value or "").strip().lower())
    return cleaned or "unknown"


def get_settings() -> AppSettings:
    return AppSettings()
