from pathlib import Path

from src.domain.models import ExperimentConfig, ForecastDriverConfig
from src.repositories.experiment_repository import ExperimentRepository


def _db_url(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'app.db'}"


def _config(experiment_id: str, owner_ldap: str, name: str = "first") -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id=experiment_id,
        owner_ldap=owner_ldap,
        name=name,
        source_file="source.xlsx",
        sheet_name="Sheet1",
        timestamp_column="date",
        target_column="amount",
        item_columns=["account"],
    )


def test_create_experiment_is_idempotent_for_existing_draft_id(tmp_path: Path) -> None:
    repository = ExperimentRepository(_db_url(tmp_path))
    config = _config("exp_repeat", "alice")

    repository.create_experiment(
        config,
        data_start="2024-01-01",
        data_end="2024-12-31",
        item_count=1,
        row_count=12,
    )
    repository.create_experiment(
        ExperimentConfig(**{**config.__dict__, "name": "second"}),
        data_start="2024-01-01",
        data_end="2025-12-31",
        item_count=2,
        row_count=24,
    )

    experiments = repository.list_experiments("alice")

    assert len(experiments) == 1
    assert experiments[0].name == "second"
    assert experiments[0].row_count == 24
    assert experiments[0].owner_ldap == "alice"


def test_create_experiment_serializes_typed_driver_configs(tmp_path: Path) -> None:
    repository = ExperimentRepository(_db_url(tmp_path))
    config = ExperimentConfig(
        experiment_id="exp_drivers",
        owner_ldap="alice",
        name="drivers",
        source_file="source.xlsx",
        sheet_name="Sheet1",
        timestamp_column="date",
        target_column="amount",
        item_columns=["account"],
        driver_configs=[
            ForecastDriverConfig(
                name="增长率",
                config_type="growth_rate",
                growth_rate=0.01,
            )
        ],
    )

    repository.create_experiment(
        config,
        data_start="2024-01-01",
        data_end="2024-12-31",
        item_count=1,
        row_count=12,
    )

    saved = repository.get_experiment("exp_drivers", "alice")

    assert saved is not None
    assert saved.config["driver_configs"] == [
        {
            "name": "增长率",
            "config_type": "growth_rate",
            "column": None,
            "availability": None,
            "future_value_strategy": None,
            "future_value_coeff": None,
            "growth_rate": 0.01,
            "impact_months": [],
            "impact_rate": None,
            "base_value": None,
            "scenario_growth_rate": None,
            "effect_rate": None,
            "enabled": True,
        }
    ]


# ── 数据隔离: repository 层是最后一道防线 ──────────────────────────


def _seed(repository: ExperimentRepository, experiment_id: str, owner: str) -> None:
    repository.create_experiment(
        _config(experiment_id, owner, name=experiment_id),
        data_start="2024-01-01",
        data_end="2024-12-31",
        item_count=1,
        row_count=12,
    )


def test_user_a_cannot_see_user_b_experiments(tmp_path: Path) -> None:
    repository = ExperimentRepository(_db_url(tmp_path))
    _seed(repository, "exp_a", "alice")
    _seed(repository, "exp_b", "bob")

    alice_view = repository.list_experiments("alice")
    bob_view = repository.list_experiments("bob")

    assert {e.id for e in alice_view} == {"exp_a"}
    assert {e.id for e in bob_view} == {"exp_b"}


def test_user_a_cannot_get_user_b_experiment_by_id(tmp_path: Path) -> None:
    repository = ExperimentRepository(_db_url(tmp_path))
    _seed(repository, "exp_b", "bob")

    # alice 用 bob 的 experiment_id 直接访问, 必须拿不到
    assert repository.get_experiment("exp_b", "alice") is None
    # bob 自己能拿到
    assert repository.get_experiment("exp_b", "bob") is not None


def test_user_a_cannot_update_user_b_experiment(tmp_path: Path) -> None:
    from src.domain.enums import ExperimentStatus

    repository = ExperimentRepository(_db_url(tmp_path))
    _seed(repository, "exp_b", "bob")

    # alice 越权更新 bob 的实验状态, 不应生效
    repository.update_status("exp_b", ExperimentStatus.FAILED, owner_ldap="alice")
    bob_exp = repository.get_experiment("exp_b", "bob")
    assert bob_exp is not None
    assert bob_exp.status != ExperimentStatus.FAILED


def test_owner_ldap_is_persisted_on_runs(tmp_path: Path) -> None:
    repository = ExperimentRepository(_db_url(tmp_path))
    _seed(repository, "exp_a", "alice")

    repository.create_run("exp_a", owner_ldap="alice", log_path="/tmp/run.log")

    # bob 看不到 alice 的 run
    assert repository.latest_run("exp_a", "bob") is None
    assert repository.latest_run("exp_a", "alice") is not None
