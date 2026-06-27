"""通用数据库层。

- 通过 SQLAlchemy 引擎统一对接 MySQL / SQLite。
- 引擎按 URL 缓存(同一个 URL 全进程共享一个连接池, MySQL 默认 pool_size=5)。
- initialize_database() 兼容 MySQL / SQLite, 表结构等价。
"""
from __future__ import annotations

from threading import Lock

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine

_engine_cache: dict[str, Engine] = {}
_engine_lock = Lock()


def get_engine(database_url: str) -> Engine:
    """按 URL 缓存引擎; 进程内重复调用复用同一连接池。"""
    with _engine_lock:
        eng = _engine_cache.get(database_url)
        if eng is not None:
            return eng

        if database_url.startswith("sqlite"):
            eng = create_engine(
                database_url,
                future=True,
                connect_args={"check_same_thread": False, "timeout": 30},
                pool_pre_ping=True,
            )
            # SQLite 也开启 WAL + busy_timeout, 避免多 worker 写竞争
            @event.listens_for(eng, "connect")
            def _sqlite_pragma(dbapi_conn, _):
                cur = dbapi_conn.cursor()
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA busy_timeout=5000")
                cur.execute("PRAGMA synchronous=NORMAL")
                cur.close()
        else:
            eng = create_engine(
                database_url,
                future=True,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
                pool_recycle=1800,
            )
        _engine_cache[database_url] = eng
        return eng


# ── 表 DDL: MySQL/SQLite 通用写法(JSON 列用 TEXT, 时间用 VARCHAR ISO8601) ──
_DDL_EXPERIMENTS = """
CREATE TABLE IF NOT EXISTS experiments (
    id VARCHAR(64) PRIMARY KEY,
    owner_ldap VARCHAR(64) NOT NULL DEFAULT '',
    name VARCHAR(255) NOT NULL,
    status VARCHAR(32) NOT NULL,
    source_file TEXT,
    sheet_name VARCHAR(128),
    target_column VARCHAR(128),
    timestamp_column VARCHAR(128),
    item_columns_json TEXT NOT NULL,
    config_json TEXT NOT NULL,
    data_start VARCHAR(32),
    data_end VARCHAR(32),
    item_count INTEGER NOT NULL DEFAULT 0,
    row_count INTEGER NOT NULL DEFAULT 0,
    best_model VARCHAR(128),
    best_wape DOUBLE,
    baseline_wape DOUBLE,
    improvement_rate DOUBLE,
    created_at VARCHAR(32) NOT NULL,
    updated_at VARCHAR(32) NOT NULL
)
"""

_DDL_RUNS = """
CREATE TABLE IF NOT EXISTS runs (
    id VARCHAR(64) PRIMARY KEY,
    experiment_id VARCHAR(64) NOT NULL,
    owner_ldap VARCHAR(64) NOT NULL DEFAULT '',
    status VARCHAR(32) NOT NULL,
    stage VARCHAR(64) NOT NULL,
    progress INTEGER NOT NULL,
    worker_pid INTEGER,
    started_at VARCHAR(32),
    ended_at VARCHAR(32),
    error_code VARCHAR(64),
    error_message TEXT,
    log_path TEXT,
    created_at VARCHAR(32) NOT NULL,
    updated_at VARCHAR(32) NOT NULL
)
"""

_DDL_INDEX_RUNS = (
    "CREATE INDEX IF NOT EXISTS idx_runs_experiment ON runs (experiment_id)",
    "CREATE INDEX IF NOT EXISTS idx_runs_status ON runs (status)",
    "CREATE INDEX IF NOT EXISTS idx_runs_owner ON runs (owner_ldap, status)",
    "CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments (status)",
    "CREATE INDEX IF NOT EXISTS idx_experiments_owner ON experiments (owner_ldap, updated_at)",
)


# 对存量库(建表早于 owner_ldap 引入)做幂等补列。
_DDL_ALTER_OWNER = (
    "ALTER TABLE experiments ADD COLUMN owner_ldap VARCHAR(64) NOT NULL DEFAULT ''",
    "ALTER TABLE runs ADD COLUMN owner_ldap VARCHAR(64) NOT NULL DEFAULT ''",
)


def initialize_database(database_url: str) -> Engine:
    """建表 + 补列 + 索引(幂等)。MySQL 8 与 SQLite 3.x 均兼容。"""
    eng = get_engine(database_url)
    is_mysql = eng.dialect.name == "mysql"
    with eng.begin() as conn:
        # MySQL 建表加 InnoDB + utf8mb4
        conn.execute(text(
            _DDL_EXPERIMENTS + (" ENGINE=InnoDB DEFAULT CHARSET=utf8mb4" if is_mysql else "")
        ))
        conn.execute(text(
            _DDL_RUNS + (" ENGINE=InnoDB DEFAULT CHARSET=utf8mb4" if is_mysql else "")
        ))
        # 存量库补 owner_ldap 列(列已存在则忽略)
        for ddl in _DDL_ALTER_OWNER:
            try:
                conn.execute(text(ddl))
            except Exception:
                pass
        for ddl in _DDL_INDEX_RUNS:
            try:
                # MySQL 不支持 IF NOT EXISTS for CREATE INDEX (8.0 之前); 用 try/except 兜底
                conn.execute(text(ddl.replace("IF NOT EXISTS ", "") if is_mysql else ddl))
            except Exception:
                pass
    return eng
