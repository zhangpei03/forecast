from pathlib import Path

from src.domain.models import ExperimentConfig, ForecastDriverConfig
from src.repositories.experiment_repository import ExperimentRepository


def test_create_experiment_is_idempotent_for_existing_draft_id(tmp_path: Path) -> None:
    repository = ExperimentRepository(tmp_path / "app.db")
    config = ExperimentConfig(
        experiment_id="exp_repeat",
        name="first",
        source_file="source.xlsx",
        sheet_name="Sheet1",
        timestamp_column="date",
        target_column="amount",
        item_columns=["account"],
    )

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

    experiments = repository.list_experiments()

    assert len(experiments) == 1
    assert experiments[0].name == "second"
    assert experiments[0].row_count == 24


def test_create_experiment_serializes_typed_driver_configs(tmp_path: Path) -> None:
    repository = ExperimentRepository(tmp_path / "app.db")
    config = ExperimentConfig(
        experiment_id="exp_drivers",
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

    saved = repository.get_experiment("exp_drivers")

    assert saved is not None
    assert saved.config["driver_configs"] == [
        {
            "name": "增长率",
            "config_type": "growth_rate",
            "column": None,
            "availability": None,
            "future_value_strategy": None,
            "growth_rate": 0.01,
            "impact_months": [],
            "impact_rate": None,
            "base_value": None,
            "scenario_growth_rate": None,
            "effect_rate": None,
            "enabled": True,
        }
    ]
