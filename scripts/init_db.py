from __future__ import annotations

from src.core.config import get_settings
from src.storage.file_store import ensure_runtime_dirs
from src.storage.sqlite import initialize_database


def main() -> None:
    settings = get_settings()
    ensure_runtime_dirs(settings)
    initialize_database(settings.database_path)
    print(f"Initialized database at {settings.database_path}")


if __name__ == "__main__":
    main()
