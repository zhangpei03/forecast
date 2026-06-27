from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text

from src.domain.enums import ExperimentStatus
from src.domain.models import ExperimentConfig, ExperimentSummary
from src.storage.db import initialize_database


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _loads_list(value: str | None) -> list[str]:
    return list(json.loads(value or "[]"))


def _row_to_summary(row: Mapping[str, Any]) -> ExperimentSummary:
    return ExperimentSummary(
        id=row["id"],
        owner_ldap=row["owner_ldap"],
        name=row["name"],
        status=ExperimentStatus(row["status"]),
        source_file=row["source_file"],
        sheet_name=row["sheet_name"],
        target_column=row["target_column"],
        timestamp_column=row["timestamp_column"],
        item_columns=_loads_list(row["item_columns_json"]),
        config=json.loads(row["config_json"] or "{}"),
        data_start=row["data_start"],
        data_end=row["data_end"],
        item_count=row["item_count"],
        row_count=row["row_count"],
        best_model=row["best_model"],
        best_wape=row["best_wape"],
        baseline_wape=row["baseline_wape"],
        improvement_rate=row["improvement_rate"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


class ExperimentRepository:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self.engine = initialize_database(database_url)

    def _conn(self):
        return self.engine.begin()  # 上下文管理器: 进事务, 退出自动 commit

    def list_experiments(self, owner_ldap: str) -> list[ExperimentSummary]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT * FROM experiments WHERE owner_ldap = :owner "
                    "ORDER BY updated_at DESC"
                ),
                {"owner": owner_ldap},
            ).mappings().all()
        return [_row_to_summary(row) for row in rows]

    def get_experiment(self, experiment_id: str, owner_ldap: str) -> ExperimentSummary | None:
        """按 (id, owner) 双条件读取; 越权访问(他人实验)返回 None。"""
        with self.engine.connect() as conn:
            row = conn.execute(
                text("SELECT * FROM experiments WHERE id = :id AND owner_ldap = :owner"),
                {"id": experiment_id, "owner": owner_ldap},
            ).mappings().first()
        return _row_to_summary(row) if row else None

    def count_active_runs(self, owner_ldap: str | None = None) -> int:
        """统计 QUEUED / RUNNING 的 run 数量, 用于并发控制。

        owner_ldap 为 None 时统计全局(容器级 CPU 保护);
        传入具体用户时只统计该用户的活跃任务。
        """
        with self.engine.connect() as conn:
            if owner_ldap is None:
                return int(
                    conn.execute(
                        text("SELECT COUNT(*) FROM runs WHERE status IN ('QUEUED','RUNNING')")
                    ).scalar()
                    or 0
                )
            return int(
                conn.execute(
                    text(
                        "SELECT COUNT(*) FROM runs "
                        "WHERE status IN ('QUEUED','RUNNING') AND owner_ldap = :owner"
                    ),
                    {"owner": owner_ldap},
                ).scalar()
                or 0
            )

    def create_experiment(
        self,
        config: ExperimentConfig,
        *,
        data_start: str | None,
        data_end: str | None,
        item_count: int,
        row_count: int,
    ) -> str:
        now = utc_now_iso()
        payload = asdict(config)
        params = {
            "id": config.experiment_id,
            "owner_ldap": config.owner_ldap,
            "name": config.name,
            "status": ExperimentStatus.VALIDATED.value,
            "source_file": config.source_file,
            "sheet_name": config.sheet_name,
            "target_column": config.target_column,
            "timestamp_column": config.timestamp_column,
            "item_columns_json": json.dumps(config.item_columns, ensure_ascii=False),
            "config_json": json.dumps(payload, ensure_ascii=False),
            "data_start": data_start,
            "data_end": data_end,
            "item_count": item_count,
            "row_count": row_count,
            "created_at": now,
            "updated_at": now,
        }
        with self._conn() as conn:
            existing = conn.execute(
                text("SELECT 1 FROM experiments WHERE id = :id"),
                {"id": config.experiment_id},
            ).first()
            if existing:
                conn.execute(
                    text(
                        """
                        UPDATE experiments SET
                            name = :name,
                            status = :status,
                            source_file = :source_file,
                            sheet_name = :sheet_name,
                            target_column = :target_column,
                            timestamp_column = :timestamp_column,
                            item_columns_json = :item_columns_json,
                            config_json = :config_json,
                            data_start = :data_start,
                            data_end = :data_end,
                            item_count = :item_count,
                            row_count = :row_count,
                            best_model = NULL,
                            best_wape = NULL,
                            baseline_wape = NULL,
                            improvement_rate = NULL,
                            updated_at = :updated_at
                        WHERE id = :id
                        """
                    ),
                    params,
                )
            else:
                conn.execute(
                    text(
                        """
                        INSERT INTO experiments (
                            id, owner_ldap, name, status, source_file, sheet_name, target_column,
                            timestamp_column, item_columns_json, config_json, data_start,
                            data_end, item_count, row_count, created_at, updated_at
                        ) VALUES (
                            :id, :owner_ldap, :name, :status, :source_file, :sheet_name, :target_column,
                            :timestamp_column, :item_columns_json, :config_json, :data_start,
                            :data_end, :item_count, :row_count, :created_at, :updated_at
                        )
                        """
                    ),
                    params,
                )
        return config.experiment_id

    def update_status(
        self, experiment_id: str, status: ExperimentStatus, *, owner_ldap: str | None = None
    ) -> None:
        sql = (
            "UPDATE experiments SET status = :status, updated_at = :updated_at "
            "WHERE id = :id"
        )
        params: dict[str, Any] = {
            "status": status.value,
            "updated_at": utc_now_iso(),
            "id": experiment_id,
        }
        if owner_ldap is not None:
            sql += " AND owner_ldap = :owner"
            params["owner"] = owner_ldap
        with self._conn() as conn:
            conn.execute(text(sql), params)

    def update_results(
        self,
        experiment_id: str,
        *,
        status: ExperimentStatus,
        best_model: str | None,
        best_wape: float | None,
        baseline_wape: float | None,
        improvement_rate: float | None,
        owner_ldap: str | None = None,
    ) -> None:
        sql = """
                    UPDATE experiments
                       SET status = :status,
                           best_model = :best_model,
                           best_wape = :best_wape,
                           baseline_wape = :baseline_wape,
                           improvement_rate = :improvement_rate,
                           updated_at = :updated_at
                     WHERE id = :id
        """
        params: dict[str, Any] = {
            "status": status.value,
            "best_model": best_model,
            "best_wape": best_wape,
            "baseline_wape": baseline_wape,
            "improvement_rate": improvement_rate,
            "updated_at": utc_now_iso(),
            "id": experiment_id,
        }
        if owner_ldap is not None:
            sql += " AND owner_ldap = :owner"
            params["owner"] = owner_ldap
        with self._conn() as conn:
            conn.execute(text(sql), params)

    def create_run(self, experiment_id: str, *, owner_ldap: str, log_path: str) -> str:
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        now = utc_now_iso()
        with self._conn() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO runs (
                        id, experiment_id, owner_ldap, status, stage, progress, log_path,
                        created_at, updated_at
                    ) VALUES (
                        :id, :experiment_id, :owner_ldap, :status, :stage, :progress, :log_path,
                        :created_at, :updated_at
                    )
                    """
                ),
                {
                    "id": run_id,
                    "experiment_id": experiment_id,
                    "owner_ldap": owner_ldap,
                    "status": "QUEUED",
                    "stage": "LOAD_DATA",
                    "progress": 0,
                    "log_path": log_path,
                    "created_at": now,
                    "updated_at": now,
                },
            )
        return run_id

    def update_run(
        self,
        run_id: str,
        *,
        status: str,
        stage: str,
        progress: int,
        worker_pid: int | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        ended_at: str | None = None,
    ) -> None:
        now = utc_now_iso()
        with self._conn() as conn:
            conn.execute(
                text(
                    """
                    UPDATE runs
                       SET status = :status,
                           stage = :stage,
                           progress = :progress,
                           worker_pid = COALESCE(:worker_pid, worker_pid),
                           error_code = :error_code,
                           error_message = :error_message,
                           ended_at = COALESCE(:ended_at, ended_at),
                           started_at = COALESCE(started_at, :started_default),
                           updated_at = :updated_at
                     WHERE id = :id
                    """
                ),
                {
                    "status": status,
                    "stage": stage,
                    "progress": progress,
                    "worker_pid": worker_pid,
                    "error_code": error_code,
                    "error_message": error_message,
                    "ended_at": ended_at,
                    "started_default": now,
                    "updated_at": now,
                    "id": run_id,
                },
            )

    def latest_run(
        self, experiment_id: str, owner_ldap: str | None = None
    ) -> dict[str, Any] | None:
        sql = "SELECT * FROM runs WHERE experiment_id = :id"
        params: dict[str, Any] = {"id": experiment_id}
        if owner_ldap is not None:
            sql += " AND owner_ldap = :owner"
            params["owner"] = owner_ldap
        sql += " ORDER BY created_at DESC LIMIT 1"
        with self.engine.connect() as conn:
            row = conn.execute(text(sql), params).mappings().first()
        return dict(row) if row else None
