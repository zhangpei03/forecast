from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.core.config import AppSettings


def ensure_runtime_dirs(settings: AppSettings) -> None:
    settings.runtime_dir.mkdir(parents=True, exist_ok=True)
    # 仅创建顶层根目录; 用户子目录由首次访问时按需创建, 避免新用户登录时全量遍历。
    settings.upload_root.mkdir(parents=True, exist_ok=True)
    settings.experiments_root.mkdir(parents=True, exist_ok=True)


def get_experiment_dir(settings: AppSettings, owner_ldap: str, experiment_id: str) -> Path:
    path = settings.experiments_dir(owner_ldap) / experiment_id
    (path / "results").mkdir(parents=True, exist_ok=True)
    (path / "models").mkdir(parents=True, exist_ok=True)
    (path / "exports").mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
