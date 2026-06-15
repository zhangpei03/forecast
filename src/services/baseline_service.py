from __future__ import annotations

import pandas as pd

from src.core.constants import ROLLING_MEAN_WINDOW, SEASONAL_LAG

BASELINE_MODELS = ("Last Value", "Seasonal Naive", "Rolling Mean")


def generate_baseline_backtest_predictions(
    *,
    data: pd.DataFrame,
    freq: str,
    prediction_length: int,
    num_windows: int,
) -> pd.DataFrame:
    clean_data = data.dropna(subset=["item_id", "timestamp", "target"]).copy()
    clean_data["timestamp"] = pd.to_datetime(clean_data["timestamp"])
    clean_data = clean_data.sort_values(["item_id", "timestamp"])
    frames: list[pd.DataFrame] = []

    for item_id, series in clean_data.groupby("item_id", sort=True):
        series = series.sort_values("timestamp").reset_index(drop=True)
        for window_index in range(num_windows):
            test_end = len(series) - window_index * prediction_length
            test_start = test_end - prediction_length
            if test_start <= 0:
                continue
            train = series.iloc[:test_start]
            actual = series.iloc[test_start:test_end]
            if train.empty or actual.empty:
                continue
            frames.extend(
                [
                    _predict_last_value(item_id, actual, train, window_index),
                    _predict_seasonal_naive(item_id, actual, train, freq, window_index),
                    _predict_rolling_mean(item_id, actual, train, freq, window_index),
                ]
            )

    if not frames:
        return pd.DataFrame(columns=_prediction_columns())
    predictions = pd.concat(frames, ignore_index=True)
    return _with_error_columns(predictions)


def generate_baseline_future_forecast(
    *,
    data: pd.DataFrame,
    freq: str,
    prediction_length: int,
    model: str,
) -> pd.DataFrame:
    clean_data = data.dropna(subset=["item_id", "timestamp", "target"]).copy()
    clean_data["timestamp"] = pd.to_datetime(clean_data["timestamp"])
    clean_data = clean_data.sort_values(["item_id", "timestamp"])
    pandas_freq = {"M": "MS", "W": "W-MON", "D": "D"}[freq]
    frames = []
    for item_id, series in clean_data.groupby("item_id", sort=True):
        series = series.sort_values("timestamp").reset_index(drop=True)
        future_dates = pd.date_range(
            series["timestamp"].iloc[-1],
            periods=prediction_length + 1,
            freq=pandas_freq,
        )[1:]
        actual = pd.DataFrame({"timestamp": future_dates, "target": [pd.NA] * prediction_length})
        if model == "Last Value":
            forecasts = [float(series["target"].iloc[-1])] * prediction_length
        elif model == "Seasonal Naive":
            lag = SEASONAL_LAG[freq]
            fallback = float(series["target"].iloc[-1])
            history = list(series["target"].astype(float))
            forecasts = []
            for horizon in range(prediction_length):
                forecasts.append(_seasonal_value(history, forecasts, lag, horizon, fallback))
        else:
            window = ROLLING_MEAN_WINDOW[freq]
            forecasts = [float(series["target"].tail(window).mean())] * prediction_length
        frame = _prediction_frame(model, item_id, actual, forecasts, -1)
        frame["window_id"] = "FUTURE"
        frame["actual"] = pd.NA
        frame["error"] = pd.NA
        frame["absolute_error"] = pd.NA
        frame["error_rate"] = pd.NA
        frames.append(frame)
    return (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=_prediction_columns())
    )


def _predict_last_value(
    item_id: str,
    actual: pd.DataFrame,
    train: pd.DataFrame,
    window_index: int,
) -> pd.DataFrame:
    forecast = float(train["target"].iloc[-1])
    return _prediction_frame("Last Value", item_id, actual, [forecast] * len(actual), window_index)


def _predict_seasonal_naive(
    item_id: str,
    actual: pd.DataFrame,
    train: pd.DataFrame,
    freq: str,
    window_index: int,
) -> pd.DataFrame:
    lag = SEASONAL_LAG[freq]
    forecasts = []
    fallback = float(train["target"].iloc[-1])
    history = list(train["target"].astype(float))
    for horizon_index in range(len(actual)):
        forecasts.append(_seasonal_value(history, forecasts, lag, horizon_index, fallback))
    return _prediction_frame("Seasonal Naive", item_id, actual, forecasts, window_index)


def _predict_rolling_mean(
    item_id: str,
    actual: pd.DataFrame,
    train: pd.DataFrame,
    freq: str,
    window_index: int,
) -> pd.DataFrame:
    window = ROLLING_MEAN_WINDOW[freq]
    forecast = float(train["target"].tail(window).mean())
    return _prediction_frame(
        "Rolling Mean", item_id, actual, [forecast] * len(actual), window_index
    )


def _prediction_frame(
    model: str,
    item_id: str,
    actual: pd.DataFrame,
    forecasts: list[float],
    window_index: int,
) -> pd.DataFrame:
    actual_values = pd.to_numeric(actual["target"], errors="coerce")
    return pd.DataFrame(
        {
            "item_id": item_id,
            "timestamp": actual["timestamp"].to_list(),
            "window_id": f"W{window_index + 1}",
            "model": model,
            "model_type": "基线",
            "actual": actual_values.to_list(),
            "forecast_mean": forecasts,
            "forecast_p10": forecasts,
            "forecast_p50": forecasts,
            "forecast_p90": forecasts,
        }
    )


def _seasonal_value(
    history: list[float],
    forecasts: list[float],
    lag: int,
    horizon_index: int,
    fallback: float,
) -> float:
    source_index = len(history) - lag + horizon_index
    if source_index < 0:
        return fallback
    if source_index < len(history):
        return float(history[source_index])
    forecast_index = source_index - len(history)
    return float(forecasts[forecast_index]) if forecast_index < len(forecasts) else fallback


def _with_error_columns(predictions: pd.DataFrame) -> pd.DataFrame:
    enriched = predictions.copy()
    enriched["error"] = enriched["forecast_p50"] - enriched["actual"]
    enriched["absolute_error"] = enriched["error"].abs()
    enriched["error_rate"] = enriched.apply(
        lambda row: row["error"] / abs(row["actual"]) if row["actual"] != 0 else pd.NA,
        axis=1,
    )
    return enriched


def _prediction_columns() -> list[str]:
    return [
        "item_id",
        "timestamp",
        "window_id",
        "model",
        "model_type",
        "actual",
        "forecast_mean",
        "forecast_p10",
        "forecast_p50",
        "forecast_p90",
        "error",
        "absolute_error",
        "error_rate",
    ]
