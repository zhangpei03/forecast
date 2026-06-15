import numpy as np
import pandas as pd

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
