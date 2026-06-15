import pandas as pd

from src.services.baseline_service import generate_baseline_backtest_predictions


def test_generate_baseline_backtest_predictions_for_monthly_series() -> None:
    dates = pd.date_range("2022-01-01", periods=30, freq="MS")
    data = pd.DataFrame(
        {
            "item_id": ["A"] * 30,
            "timestamp": dates,
            "target": [float(index + 1) for index in range(30)],
        }
    )

    predictions = generate_baseline_backtest_predictions(
        data=data,
        freq="M",
        prediction_length=3,
        num_windows=2,
    )

    last_value = predictions[
        (predictions["model"] == "Last Value") & (predictions["window_id"] == "W1")
    ].sort_values("timestamp")
    seasonal = predictions[
        (predictions["model"] == "Seasonal Naive") & (predictions["window_id"] == "W1")
    ].sort_values("timestamp")
    rolling = predictions[
        (predictions["model"] == "Rolling Mean") & (predictions["window_id"] == "W1")
    ].sort_values("timestamp")

    assert list(last_value["forecast_p50"]) == [27.0, 27.0, 27.0]
    assert list(seasonal["forecast_p50"]) == [16.0, 17.0, 18.0]
    assert list(rolling["forecast_p50"]) == [26.0, 26.0, 26.0]
    assert set(predictions["model"]) == {"Last Value", "Seasonal Naive", "Rolling Mean"}


def test_daily_seasonal_naive_reuses_prior_forecasts_when_horizon_exceeds_lag() -> None:
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    data = pd.DataFrame(
        {
            "item_id": ["A"] * len(dates),
            "timestamp": dates,
            "target": [float(index + 1) for index in range(len(dates))],
        }
    )

    predictions = generate_baseline_backtest_predictions(
        data=data,
        freq="D",
        prediction_length=30,
        num_windows=1,
    )

    seasonal = predictions[predictions["model"] == "Seasonal Naive"].sort_values("timestamp")

    assert len(seasonal) == 30
    assert list(seasonal["forecast_p50"].head(7)) == [84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0]
    assert list(seasonal["forecast_p50"].iloc[7:14]) == [84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0]
