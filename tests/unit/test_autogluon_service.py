import pandas as pd

from src.services.autogluon_service import (
    _align_known_covariates_to_future,
    _format_autogluon_predictions,
    _lightweight_hyperparameters,
    hyperparameters_for_preset,
)

_DEEP_MODELS = {"DeepAR", "TemporalFusionTransformer", "PatchTST", "Chronos2"}


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
