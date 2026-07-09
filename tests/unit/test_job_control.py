from pathlib import Path

from src.core.config import AppSettings
from src.domain.enums import ExperimentStatus
from src.domain.models import ExperimentConfig
from src.jobs.job_control import cancel_training_job
from src.repositories.experiment_repository import ExperimentRepository
from src.storage.file_store import get_experiment_dir, read_json, write_json


def _db_url(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'app.db'}"


def _config(experiment_id: str, owner_ldap: str) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id=experiment_id,
        owner_ldap=owner_ldap,
        name="cancel test",
        source_file="source.xlsx",
        sheet_name="Sheet1",
        timestamp_column="date",
        target_column="amount",
        item_columns=["account"],
    )


def test_cancel_training_job_marks_run_cancelled_and_releases_active_slot(tmp_path: Path) -> None:
    settings = AppSettings(runtime_dir=tmp_path / "runtime", database_url=_db_url(tmp_path))
    repository = ExperimentRepository(settings.database_url)
    experiment_id = "exp_cancel"
    owner_ldap = "alice"

    repository.create_experiment(
        _config(experiment_id, owner_ldap),
        data_start="2024-01-01",
        data_end="2024-12-31",
        item_count=1,
        row_count=12,
    )
    run_id = repository.create_run(
        experiment_id,
        owner_ldap=owner_ldap,
        log_path=str(tmp_path / "runtime" / "run.log"),
    )
    repository.update_status(experiment_id, ExperimentStatus.RUNNING, owner_ldap=owner_ldap)
    repository.update_run(
        run_id,
        status=ExperimentStatus.RUNNING.value,
        stage="TRAIN_AUTOGLUON",
        progress=42,
    )
    experiment_dir = get_experiment_dir(settings, owner_ldap, experiment_id)
    write_json(
        experiment_dir / "progress.json",
        {
            "experiment_id": experiment_id,
            "status": ExperimentStatus.RUNNING.value,
            "stage": "TRAIN_AUTOGLUON",
            "progress": 42,
            "message": "training",
            "worker_pid": None,
            "log_path": str(experiment_dir / "run.log"),
        },
    )

    assert repository.count_active_runs() == 1

    result = cancel_training_job(
        settings=settings,
        repository=repository,
        experiment_id=experiment_id,
        owner_ldap=owner_ldap,
    )

    assert result.status_changed is True
    assert result.process_terminated is False
    assert repository.count_active_runs() == 0

    latest_run = repository.latest_run(experiment_id, owner_ldap)
    assert latest_run is not None
    assert latest_run["status"] == ExperimentStatus.CANCELLED.value
    assert latest_run["error_code"] == "USER_CANCELLED"
    assert latest_run["ended_at"] is not None

    summary = repository.get_experiment(experiment_id, owner_ldap)
    assert summary is not None
    assert summary.status == ExperimentStatus.CANCELLED

    progress = read_json(experiment_dir / "progress.json")
    assert progress["status"] == ExperimentStatus.CANCELLED.value
    assert progress["message"] == "训练已手动停止"
