import pandas as pd

from src.domain.models import ExperimentConfig, ForecastDriverConfig
from src.services.autogluon_service import (
    _align_known_covariates_to_future,
    _autogluon_covariate_columns,
    _format_autogluon_predictions,
    _lightweight_hyperparameters,
    _static_features_frame,
    _time_series_input_columns,
    hyperparameters_for_preset,
)

_DEEP_MODELS = {"DeepAR", "TemporalFusionTransformer", "PatchTST", "Chronos2"}


def _config(**overrides) -> ExperimentConfig:
    payload = {
        "experiment_id": "exp_test",
        "name": "test",
        "source_file": "source.xlsx",
        "sheet_name": "Sheet1",
        "timestamp_column": "date",
        "target_column": "gmv",
        "item_columns": ["city"],
    }
    payload.update(overrides)
    return ExperimentConfig(**payload)


def test_lightweight_hyperparameters_excludes_heavy_deep_models() -> None:
    hyperparameters = _lightweight_hyperparameters()

    assert set(hyperparameters) == {
        "SeasonalNaive",
        "RecursiveTabular",
        "DirectTabular",
        "ETS",
        "Theta",
    }
    assert "Chronos2" not in hyperparameters
    assert "TemporalFusionTransformer" not in hyperparameters


def test_fast_preset_uses_lightweight_models_only() -> None:
    hyperparameters = hyperparameters_for_preset("fast_training")

    assert hyperparameters == _lightweight_hyperparameters()
    assert _DEEP_MODELS.isdisjoint(hyperparameters)


def test_medium_preset_adds_lightgbm_backend_without_deep_models() -> None:
    hyperparameters = hyperparameters_for_preset("medium_quality")

    assert hyperparameters["RecursiveTabular"] == {"model_name": "GBM"}
    assert hyperparameters["DirectTabular"] == {"model_name": "GBM"}
    assert _DEEP_MODELS.isdisjoint(hyperparameters)


def test_high_preset_enables_deep_learning_models() -> None:
    hyperparameters = hyperparameters_for_preset("high_quality")

    assert _DEEP_MODELS.issubset(hyperparameters)
    assert hyperparameters["Chronos2"] == {"model_path": "autogluon/chronos-2"}
    assert {"SeasonalNaive", "ETS", "Theta"}.issubset(hyperparameters)


def test_hyperparameters_for_preset_can_select_single_model() -> None:
    assert hyperparameters_for_preset("medium_quality", "Theta") == {"Theta": {}}
    assert hyperparameters_for_preset("fast_training", "Chronos") == {
        "Chronos2": {"model_path": "autogluon/chronos-2"}
    }


def test_hyperparameters_for_preset_can_select_multiple_models() -> None:
    assert hyperparameters_for_preset("medium_quality", ["Theta", "DirectTabular"]) == {
        "Theta": {},
        "DirectTabular": {"model_name": "GBM"},
    }


def test_autogluon_covariate_columns_split_known_and_past_inputs() -> None:
    data = pd.DataFrame(
        {
            "item_id": ["A", "A"],
            "timestamp": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "target": [100.0, 110.0],
            "weekday": [4, 5],
            "tsh": [10.0, 11.0],
            "missing_from_data": [1.0, 2.0],
        }
    ).drop(columns=["missing_from_data"])
    config = _config(
        known_covariates=["weekday"],
        past_covariates=["tsh"],
        driver_configs=[
            ForecastDriverConfig(
                name="weekday",
                config_type="covariate",
                column="weekday",
                availability="known_future",
            ),
            ForecastDriverConfig(
                name="tsh",
                config_type="covariate",
                column="tsh",
                availability="historical",
            ),
            ForecastDriverConfig(
                name="missing",
                config_type="covariate",
                column="missing_from_data",
                availability="historical",
            ),
        ],
    )

    known_covariates, past_covariates = _autogluon_covariate_columns(config, data)

    assert known_covariates == ["weekday"]
    assert past_covariates == ["tsh"]
    assert _time_series_input_columns(data, known_covariates, past_covariates) == [
        "item_id",
        "timestamp",
        "target",
        "weekday",
        "tsh",
    ]


def test_static_features_frame_uses_one_row_per_item() -> None:
    data = pd.DataFrame(
        {
            "item_id": ["A", "A", "B"],
            "timestamp": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-01"]),
            "target": [100.0, 110.0, 80.0],
            "city_tier": ["T1", "T1", "T2"],
            "unused": [1, 2, 3],
        }
    )

    static_features = _static_features_frame(data, ["city_tier", "missing"])

    assert static_features is not None
    assert static_features.index.tolist() == ["A", "B"]
    assert static_features["city_tier"].tolist() == ["T1", "T2"]


def test_align_known_covariates_to_future_uses_series_horizon_order() -> None:
    future_index = pd.DataFrame(
        {
            "item_id": ["A", "A", "B", "B"],
            "timestamp": pd.to_datetime(["2025-07-31", "2025-08-31", "2025-07-31", "2025-08-31"]),
        }
    )
    source = pd.DataFrame(
        {
            "item_id": ["A", "A", "B", "B"],
            "timestamp": pd.to_datetime(["2025-07-01", "2025-08-01", "2025-07-01", "2025-08-01"]),
            "workdays": [23.0, 21.0, 22.0, 20.0],
        }
    )

    aligned = _align_known_covariates_to_future(future_index, source, ["workdays"])

    assert aligned["timestamp"].tolist() == future_index["timestamp"].tolist()
    assert aligned["workdays"].tolist() == [23.0, 21.0, 22.0, 20.0]


def test_format_autogluon_predictions_aligns_monthly_actuals_by_horizon_order() -> None:
    raw_predictions = pd.DataFrame(
        {
            "mean": [110.0, 121.0],
            "0.1": [100.0, 110.0],
            "0.5": [110.0, 121.0],
            "0.9": [120.0, 132.0],
        },
        index=pd.MultiIndex.from_arrays(
            [["A", "A"], pd.to_datetime(["2025-07-31", "2025-08-31"])],
            names=["item_id", "timestamp"],
        ),
    )
    actual = pd.DataFrame(
        {
            "item_id": ["A", "A"],
            "timestamp": pd.to_datetime(["2025-07-01", "2025-08-01"]),
            "target": [110.0, 121.0],
        }
    )

    formatted = _format_autogluon_predictions(raw_predictions, actual, "W1")

    assert formatted["actual"].tolist() == [110.0, 121.0]
    assert formatted["error"].tolist() == [0.0, 0.0]
