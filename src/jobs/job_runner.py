from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from src.core.config import AppSettings
from src.domain.enums import ExperimentStatus
from src.repositories.experiment_repository import ExperimentRepository
from src.storage.file_store import get_experiment_dir, write_json


def start_training_job(
    *,
    settings: AppSettings,
    repository: ExperimentRepository,
    experiment_id: str,
) -> int:
    experiment_dir = get_experiment_dir(settings, experiment_id)
    log_path = experiment_dir / "run.log"
    run_id = repository.create_run(experiment_id, log_path=str(log_path))
    repository.update_status(experiment_id, ExperimentStatus.QUEUED)
    write_json(
        experiment_dir / "progress.json",
        {
            "experiment_id": experiment_id,
            "status": "QUEUED",
            "stage": "LOAD_DATA",
            "progress": 0,
            "message": "任务已进入本地队列",
            "started_at": None,
            "updated_at": "",
            "worker_pid": None,
            "log_path": str(log_path),
        },
    )

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "src.jobs.train_worker",
            "--experiment-id",
            experiment_id,
            "--run-id",
            run_id,
            "--database-path",
            str(settings.database_path),
            "--runtime-dir",
            str(settings.runtime_dir),
        ],
        cwd=str(Path.cwd()),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return int(process.pid)
