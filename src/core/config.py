from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.core.constants import RUNTIME_DIR


@dataclass(frozen=True)
class AppSettings:
    runtime_dir: Path = Path(RUNTIME_DIR)

    @property
    def database_path(self) -> Path:
        return self.runtime_dir / "app.db"

    @property
    def upload_dir(self) -> Path:
        return self.runtime_dir / "uploads"

    @property
    def experiments_dir(self) -> Path:
        return self.runtime_dir / "experiments"


def get_settings() -> AppSettings:
    return AppSettings()
