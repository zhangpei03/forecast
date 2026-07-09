from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import psutil

from src.core.config import AppSettings
from src.domain.enums import ExperimentStatus
from src.repositories.experiment_repository import ExperimentRepository, utc_now_iso
from src.storage.file_store import get_experiment_dir, read_json, write_json

_ACTIVE_STATUSES = {ExperimentStatus.QUEUED.value, ExperimentStatus.RUNNING.value}


@dataclass(frozen=True)
class CancelTrainingResult:
    status_changed: bool
    process_terminated: bool
    message: str


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _matches_training_worker(
    process: psutil.Process,
    *,
    experiment_id: str,
    run_id: str,
    owner_ldap: str,
) -> bool:
    try:
        cmdline = process.cmdline()
    except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
        return False

    cmdline_text = "\n".join(cmdline)
    return (
        "src.jobs.train_worker" in cmdline_text
        and "--experiment-id" in cmdline
        and experiment_id in cmdline
        and "--run-id" in cmdline
        and run_id in cmdline
        and "--owner-ldap" in cmdline
        and owner_ldap in cmdline
    )


def _terminate_worker_tree(
    pid: int | None,
    *,
    experiment_id: str,
    run_id: str,
    owner_ldap: str,
) -> tuple[bool, str]:
    if pid is None:
        return False, "未记录训练进程 PID，已只取消任务状态"

    try:
        process = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return False, "训练进程已不存在，已取消任务状态"

    if not _matches_training_worker(
        process,
        experiment_id=experiment_id,
        run_id=run_id,
        owner_ldap=owner_ldap,
    ):
        return False, "PID 已被其他进程复用，未终止进程，仅取消任务状态"

    try:
        processes = process.children(recursive=True) + [process]
        for item in processes:
            try:
                item.terminate()
            except psutil.NoSuchProcess:
                pass

        _, alive = psutil.wait_procs(processes, timeout=5)
        for item in alive:
            try:
                item.kill()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(alive, timeout=3)
    except psutil.AccessDenied:
        return False, "没有权限终止训练进程，已取消任务状态"

    return True, "已停止训练进程并取消任务"


def cancel_training_job(
    *,
    settings: AppSettings,
    repository: ExperimentRepository,
    experiment_id: str,
    owner_ldap: str,
) -> CancelTrainingResult:
    summary = repository.get_experiment(experiment_id, owner_ldap)
    if summary is None:
        return CancelTrainingResult(False, False, "实验不存在或无权访问")

    run = repository.latest_run(experiment_id, owner_ldap)
    run_is_active = bool(run and str(run.get("status")) in _ACTIVE_STATUSES)
    experiment_is_active = summary.status.value in _ACTIVE_STATUSES
    if not run_is_active and not experiment_is_active:
        return CancelTrainingResult(False, False, "当前任务不在排队或运行中")

    experiment_dir = get_experiment_dir(settings, owner_ldap, experiment_id)
    progress_path = experiment_dir / "progress.json"
    progress_payload = read_json(progress_path)
    pid = _safe_int((run or {}).get("worker_pid")) or _safe_int(
        progress_payload.get("worker_pid")
    )

    process_terminated = False
    process_message = "未找到可终止的训练进程"
    if run:
        process_terminated, process_message = _terminate_worker_tree(
            pid,
            experiment_id=experiment_id,
            run_id=str(run["id"]),
            owner_ldap=owner_ldap,
        )

    stage = str(progress_payload.get("stage") or (run or {}).get("stage") or "LOAD_DATA")
    progress = (
        _safe_int(progress_payload.get("progress"))
        or _safe_int((run or {}).get("progress"))
        or 0
    )
    ended_at = utc_now_iso()

    if run:
        repository.update_run(
            str(run["id"]),
            status=ExperimentStatus.CANCELLED.value,
            stage=stage,
            progress=progress,
            error_code="USER_CANCELLED",
            error_message="用户手动停止训练",
            ended_at=ended_at,
        )
    repository.update_status(experiment_id, ExperimentStatus.CANCELLED, owner_ldap=owner_ldap)

    progress_payload.update(
        {
            "experiment_id": experiment_id,
            "status": ExperimentStatus.CANCELLED.value,
            "stage": stage,
            "progress": progress,
            "message": "训练已手动停止",
            "updated_at": ended_at,
            "worker_pid": pid,
            "log_path": progress_payload.get("log_path") or (run or {}).get("log_path"),
        }
    )
    write_json(progress_path, progress_payload)

    return CancelTrainingResult(True, process_terminated, process_message)
