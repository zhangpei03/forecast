from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from src.core.constants import DEFAULT_RANDOM_SEED, QUANTILE_LEVELS
from src.domain.enums import ExperimentStatus


@dataclass(frozen=True)
class ForecastDriverConfig:
    """A typed business assumption or model input used by the forecasting run."""

    name: str
    config_type: str
    column: str | None = None
    availability: str | None = None
    future_value_strategy: str | None = None
    growth_rate: float | None = None
    impact_months: list[int] = field(default_factory=list)
    impact_rate: float | None = None
    base_value: float | None = None
    scenario_growth_rate: float | None = None
    effect_rate: float | None = None
    enabled: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    name: str
    source_file: str
    sheet_name: str
    timestamp_column: str
    target_column: str
    item_columns: list[str]
    known_covariates: list[str] = field(default_factory=list)
    past_covariates: list[str] = field(default_factory=list)
    static_features: list[str] = field(default_factory=list)
    driver_configs: list[ForecastDriverConfig | dict[str, Any]] = field(default_factory=list)
    freq: str = "M"
    prediction_length: int = 3
    num_val_windows: int = 3
    val_step_size: int = 3
    preset: str = "medium_quality"
    time_limit_seconds: int = 1800
    quantile_levels: list[float] = field(default_factory=lambda: QUANTILE_LEVELS.copy())
    random_seed: int = DEFAULT_RANDOM_SEED
    duplicate_strategy: str = "sum"
    missing_strategy: str = "block"


@dataclass(frozen=True)
class ExperimentSummary:
    id: str
    name: str
    status: ExperimentStatus
    source_file: str | None
    sheet_name: str | None
    target_column: str | None
    timestamp_column: str | None
    item_columns: list[str]
    config: dict[str, Any]
    data_start: str | None
    data_end: str | None
    item_count: int
    row_count: int
    best_model: str | None
    best_wape: float | None
    baseline_wape: float | None
    improvement_rate: float | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class QualityIssue:
    issue_type: str
    severity: str
    message: str
    affected_count: int
    sample: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DataProfile:
    row_count: int
    item_count: int
    data_start: str | None
    data_end: str | None
    average_series_length: float
    frequency: str
    blocking_issue_count: int
    warning_count: int
    issues: list[QualityIssue]


@dataclass(frozen=True)
class PredictionMetrics:
    wape: float | None
    mae: float | None
    rmse: float | None
    bias_amount: float
    bias_rate: float | None
    coverage: float | None


@dataclass(frozen=True)
class ProgressState:
    experiment_id: str
    status: str
    stage: str
    progress: int
    message: str
    started_at: str | None
    updated_at: str
    worker_pid: int | None = None
    log_path: str | None = None


def experiment_path(runtime_dir: Path, experiment_id: str) -> Path:
    return runtime_dir / "experiments" / experiment_id
