"""向后兼容 shim。新代码请使用 src.storage.db。

保留这个模块只是为了不破坏可能存在的外部 import。
"""
from __future__ import annotations

from src.storage.db import get_engine, initialize_database  # noqa: F401
