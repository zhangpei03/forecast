import numpy as np
import pandas as pd

from src.domain.models import ForecastDriverConfig
from src.services.custom_model_service import generate_custom_model_backtest_predictions


def test_custom_models_generate_backtest_predictions() -> None:
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    data = pd.DataFrame(
        {
            "item_id": ["A"] * len(dates),
            "timestamp": dates,
            "target": 1000 + np.arange(len(dates)) * 2 + np.sin(np.arange(len(dates))) * 5,
        }
    )

    predictions, failures = generate_custom_model_backtest_predictions(
        data=data,
        freq="D",
        prediction_length=7,
        num_windows=1,
    )

    assert failures == []
    assert set(predictions["model"]) == {"AutoARIMA", "Prophet", "XGBoost"}
    assert predictions.groupby("model").size().to_dict() == {
        "AutoARIMA": 7,
        "Prophet": 7,
        "XGBoost": 7,
    }


def test_custom_models_accept_known_future_covariates() -> None:
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    data = pd.DataFrame(
        {
            "item_id": ["A"] * len(dates),
            "timestamp": dates,
            "target": 1000 + np.arange(len(dates)) * 2 + np.sin(np.arange(len(dates))) * 5,
            "workdays": 18 + (np.arange(len(dates)) % 5),
        }
    )
    drivers = [
        ForecastDriverConfig(
            name="工作日数",
            config_type="covariate",
            column="workdays",
            availability="known_future",
            future_value_strategy="last_value",
        )
    ]

    predictions, failures = generate_custom_model_backtest_predictions(
        data=data,
        freq="D",
        prediction_length=7,
        num_windows=1,
        driver_configs=drivers,
    )

    assert failures == []
    assert predictions.groupby("model").size().to_dict() == {
        "AutoARIMA": 7,
        "Prophet": 7,
        "XGBoost": 7,
    }
