from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.domain.enums import ExperimentStatus
from src.domain.models import ExperimentConfig, ExperimentSummary
from src.storage.sqlite import connect, initialize_database


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _loads_list(value: str | None) -> list[str]:
    return list(json.loads(value or "[]"))


def _row_to_summary(row: sqlite3.Row) -> ExperimentSummary:
    return ExperimentSummary(
        id=row["id"],
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
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        initialize_database(database_path)

    def list_experiments(self) -> list[ExperimentSummary]:
        with connect(self.database_path) as connection:
            rows = connection.execute(
                "SELECT * FROM experiments ORDER BY updated_at DESC"
            ).fetchall()
        return [_row_to_summary(row) for row in rows]

    def get_experiment(self, experiment_id: str) -> ExperimentSummary | None:
        with connect(self.database_path) as connection:
            row = connection.execute(
                "SELECT * FROM experiments WHERE id = ?",
                (experiment_id,),
            ).fetchone()
        return _row_to_summary(row) if row else None

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
        payload = config.__dict__.copy()
        with connect(self.database_path) as connection:
            connection.execute(
                """
                INSERT INTO experiments (
                    id, name, status, source_file, sheet_name, target_column,
                    timestamp_column, item_columns_json, config_json, data_start,
                    data_end, item_count, row_count, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    status = excluded.status,
                    source_file = excluded.source_file,
                    sheet_name = excluded.sheet_name,
                    target_column = excluded.target_column,
                    timestamp_column = excluded.timestamp_column,
                    item_columns_json = excluded.item_columns_json,
                    config_json = excluded.config_json,
                    data_start = excluded.data_start,
                    data_end = excluded.data_end,
                    item_count = excluded.item_count,
                    row_count = excluded.row_count,
                    best_model = NULL,
                    best_wape = NULL,
                    baseline_wape = NULL,
                    improvement_rate = NULL,
                    updated_at = excluded.updated_at
                """,
                (
                    config.experiment_id,
                    config.name,
                    ExperimentStatus.VALIDATED.value,
                    config.source_file,
                    config.sheet_name,
                    config.target_column,
                    config.timestamp_column,
                    json.dumps(config.item_columns, ensure_ascii=False),
                    json.dumps(payload, ensure_ascii=False),
                    data_start,
                    data_end,
                    item_count,
                    row_count,
                    now,
                    now,
                ),
            )
        return config.experiment_id

    def update_status(self, experiment_id: str, status: ExperimentStatus) -> None:
        with connect(self.database_path) as connection:
            connection.execute(
                "UPDATE experiments SET status = ?, updated_at = ? WHERE id = ?",
                (status.value, utc_now_iso(), experiment_id),
            )

    def update_results(
        self,
        experiment_id: str,
        *,
        status: ExperimentStatus,
        best_model: str | None,
        best_wape: float | None,
        baseline_wape: float | None,
        improvement_rate: float | None,
    ) -> None:
        with connect(self.database_path) as connection:
            connection.execute(
                """
                UPDATE experiments
                   SET status = ?,
                       best_model = ?,
                       best_wape = ?,
                       baseline_wape = ?,
                       improvement_rate = ?,
                       updated_at = ?
                 WHERE id = ?
                """,
                (
                    status.value,
                    best_model,
                    best_wape,
                    baseline_wape,
                    improvement_rate,
                    utc_now_iso(),
                    experiment_id,
                ),
            )

    def create_run(self, experiment_id: str, *, log_path: str) -> str:
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        now = utc_now_iso()
        with connect(self.database_path) as connection:
            connection.execute(
                """
                INSERT INTO runs (
                    id, experiment_id, status, stage, progress, log_path,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, experiment_id, "QUEUED", "LOAD_DATA", 0, log_path, now, now),
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
        with connect(self.database_path) as connection:
            connection.execute(
                """
                UPDATE runs
                   SET status = ?,
                       stage = ?,
                       progress = ?,
                       worker_pid = COALESCE(?, worker_pid),
                       error_code = ?,
                       error_message = ?,
                       ended_at = COALESCE(?, ended_at),
                       started_at = COALESCE(started_at, ?),
                       updated_at = ?
                 WHERE id = ?
                """,
                (
                    status,
                    stage,
                    progress,
                    worker_pid,
                    error_code,
                    error_message,
                    ended_at,
                    utc_now_iso(),
                    utc_now_iso(),
                    run_id,
                ),
            )

    def latest_run(self, experiment_id: str) -> dict[str, Any] | None:
        with connect(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT * FROM runs
                 WHERE experiment_id = ?
                 ORDER BY created_at DESC
                 LIMIT 1
                """,
                (experiment_id,),
            ).fetchone()
        return dict(row) if row else None
