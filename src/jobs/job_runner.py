from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from src.core.config import AppSettings
from src.core.exceptions import ForecastLabError
from src.domain.enums import ExperimentStatus
from src.repositories.experiment_repository import ExperimentRepository
from src.storage.file_store import get_experiment_dir, write_json


class TooManyActiveJobsError(ForecastLabError):
    """活跃训练任务数已达上限。"""


def start_training_job(
    *,
    settings: AppSettings,
    repository: ExperimentRepository,
    experiment_id: str,
    owner_ldap: str,
) -> int:
    # 并发上限: 防止多用户同时提交导致 CPU/内存雪崩 + 死锁概率指数上升。
    # 这里按容器全局统计(None), 保护单容器 CPU; 不按用户隔离并发额度。
    active = repository.count_active_runs()
    if active >= settings.max_concurrent_training:
        raise TooManyActiveJobsError(
            f"当前已有 {active} 个训练任务在排队/运行, "
            f"超过上限 {settings.max_concurrent_training}, 请稍后再试。"
        )

    experiment_dir = get_experiment_dir(settings, owner_ldap, experiment_id)
    log_path = experiment_dir / "run.log"
    run_id = repository.create_run(experiment_id, owner_ldap=owner_ldap, log_path=str(log_path))
    repository.update_status(experiment_id, ExperimentStatus.QUEUED, owner_ldap=owner_ldap)
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
            "--owner-ldap",
            owner_ldap,
            "--database-url",
            settings.database_url,
            "--runtime-dir",
            str(settings.runtime_dir),
        ],
        cwd=str(Path.cwd()),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return int(process.pid)
