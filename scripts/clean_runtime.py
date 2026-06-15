from __future__ import annotations

import shutil

from src.core.config import get_settings


def main() -> None:
    runtime_dir = get_settings().runtime_dir
    if runtime_dir.exists():
        shutil.rmtree(runtime_dir)
    print(f"Removed {runtime_dir}")


if __name__ == "__main__":
    main()
