from __future__ import annotations

import contextlib
import io
import logging
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.core.constants import DEFAULT_RANDOM_SEED, SEASONAL_LAG
from src.domain.models import ForecastDriverConfig
from src.services.forecast_driver_service import (
    build_future_known_covariates,
    historical_covariate_columns,
    known_covariate_columns,
    normalize_driver_configs,
)

CUSTOM_MODEL_NAMES = ("AutoARIMA", "Prophet", "XGBoost")


@dataclass(frozen=True)
class ModelFailure:
    model: str
    message: str


def generate_custom_model_backtest_predictions(
    *,
    data: pd.DataFrame,
    freq: str,
    prediction_length: int,
    num_windows: int,
    driver_configs: Iterable[ForecastDriverConfig | Mapping[str, object]] | None = None,
) -> tuple[pd.DataFrame, list[ModelFailure]]:
    failures: list[ModelFailure] = []
    frames: list[pd.DataFrame] = []
    configs = normalize_driver_configs(driver_configs)

    for window_index in range(num_windows):
        train_data, actual_data = _split_window(data, prediction_length, window_index)
        if actual_data.empty:
            continue
        for model_name, generator in (
            ("AutoARIMA", _predict_auto_arima),
            ("Prophet", _predict_prophet),
            ("XGBoost", _predict_xgboost),
        ):
            try:
                predictions = generator(
                    train_data=train_data,
                    actual_data=actual_data,
                    freq=freq,
                    prediction_length=prediction_length,
                    window_id=f"W{window_index + 1}",
                    driver_configs=configs,
                )
                if not predictions.empty:
                    frames.append(predictions)
            except Exception as exc:
                failures.append(ModelFailure(model_name, str(exc)))

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=_columns())
    return result, failures


def generate_custom_model_future_forecast(
    *,
    data: pd.DataFrame,
    freq: str,
    prediction_length: int,
    model: str,
    driver_configs: Iterable[ForecastDriverConfig | Mapping[str, object]] | None = None,
) -> pd.DataFrame:
    configs = normalize_driver_configs(driver_configs)
    actual_data = _future_frame(data, freq, prediction_length, configs)
    if model == "AutoARIMA":
        return _predict_auto_arima(
            train_data=data,
            actual_data=actual_data,
            freq=freq,
            prediction_length=prediction_length,
            window_id="FUTURE",
            driver_configs=configs,
        )
    if model == "Prophet":
        return _predict_prophet(
            train_data=data,
            actual_data=actual_data,
            freq=freq,
            prediction_length=prediction_length,
            window_id="FUTURE",
            driver_configs=configs,
        )
    if model == "XGBoost":
        return _predict_xgboost(
            train_data=data,
            actual_data=actual_data,
            freq=freq,
            prediction_length=prediction_length,
            window_id="FUTURE",
            driver_configs=configs,
        )
    return pd.DataFrame(columns=_columns())


def _split_window(
    data: pd.DataFrame,
    prediction_length: int,
    window_index: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_frames = []
    actual_frames = []
    clean_data = data.dropna(subset=["item_id", "timestamp", "target"]).copy()
    clean_data["timestamp"] = pd.to_datetime(clean_data["timestamp"])
    clean_data = clean_data.sort_values(["item_id", "timestamp"])
    for _, series in clean_data.groupby("item_id", sort=True):
        test_end = len(series) - window_index * prediction_length
        test_start = test_end - prediction_length
        if test_start <= 0:
            continue
        train_frames.append(series.iloc[:test_start])
        actual_frames.append(series.iloc[test_start:test_end])
    train_data = pd.concat(train_frames, ignore_index=True) if train_frames else pd.DataFrame()
    actual_data = pd.concat(actual_frames, ignore_index=True) if actual_frames else pd.DataFrame()
    return train_data, actual_data


def _predict_auto_arima(
    *,
    train_data: pd.DataFrame,
    actual_data: pd.DataFrame,
    freq: str,
    prediction_length: int,
    window_id: str,
    driver_configs: Iterable[ForecastDriverConfig | Mapping[str, object]] | None = None,
) -> pd.DataFrame:
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA

    stats_freq = _statsforecast_freq(freq)
    forecast_input = train_data.rename(
        columns={"item_id": "unique_id", "timestamp": "ds", "target": "y"}
    )[["unique_id", "ds", "y"]]
    model = StatsForecast(
        models=[AutoARIMA(season_length=_season_length(freq), stepwise=True)],
        freq=stats_freq,
        n_jobs=1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        forecast = model.forecast(df=forecast_input, h=prediction_length, level=[80]).reset_index()
    forecast = forecast.rename(
        columns={
            "unique_id": "item_id",
            "ds": "timestamp",
            "AutoARIMA": "forecast_p50",
            "AutoARIMA-lo-80": "forecast_p10",
            "AutoARIMA-hi-80": "forecast_p90",
        }
    )
    forecast["forecast_mean"] = forecast["forecast_p50"]
    return _join_actuals(forecast, actual_data, "AutoARIMA", window_id)


def _predict_prophet(
    *,
    train_data: pd.DataFrame,
    actual_data: pd.DataFrame,
    freq: str,
    prediction_length: int,
    window_id: str,
    driver_configs: Iterable[ForecastDriverConfig | Mapping[str, object]] | None = None,
) -> pd.DataFrame:
    from prophet import Prophet

    frames = []
    prophet_freq = _pandas_freq(freq)
    known_covariates = _available_covariates(train_data, known_covariate_columns(driver_configs))
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    for item_id, series in train_data.groupby("item_id", sort=True):
        prophet_data = series.rename(columns={"timestamp": "ds", "target": "y"})[
            ["ds", "y", *known_covariates]
        ].dropna()
        if prophet_data["y"].nunique() <= 1:
            continue
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            model = Prophet(
                interval_width=0.8,
                daily_seasonality=freq == "D",
                weekly_seasonality=freq in {"D", "W"},
                yearly_seasonality=True,
            )
            for column in known_covariates:
                model.add_regressor(column)
            model.fit(prophet_data)
        future = pd.DataFrame(
            {
                "ds": pd.date_range(
                    pd.to_datetime(series["timestamp"]).max(),
                    periods=prediction_length + 1,
                    freq=prophet_freq,
                )[1:]
            }
        )
        if known_covariates:
            future_values = actual_data[actual_data["item_id"].eq(item_id)].copy()
            future_values["timestamp"] = pd.to_datetime(future_values["timestamp"])
            future = future.merge(
                future_values[["timestamp", *known_covariates]].rename(columns={"timestamp": "ds"}),
                on="ds",
                how="left",
            )
            for column in known_covariates:
                fallback = float(series[column].dropna().iloc[-1])
                future[column] = pd.to_numeric(future[column], errors="coerce").fillna(fallback)
        forecast = model.predict(future)
        frames.append(
            pd.DataFrame(
                {
                    "item_id": item_id,
                    "timestamp": forecast["ds"],
                    "forecast_mean": forecast["yhat"],
                    "forecast_p50": forecast["yhat"],
                    "forecast_p10": forecast["yhat_lower"],
                    "forecast_p90": forecast["yhat_upper"],
                }
            )
        )
    forecast_data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return _join_actuals(forecast_data, actual_data, "Prophet", window_id)


def _predict_xgboost(
    *,
    train_data: pd.DataFrame,
    actual_data: pd.DataFrame,
    freq: str,
    prediction_length: int,
    window_id: str,
    driver_configs: Iterable[ForecastDriverConfig | Mapping[str, object]] | None = None,
) -> pd.DataFrame:
    from xgboost import XGBRegressor

    frames = []
    lag_values = _xgboost_lags(freq)
    known_covariates = _available_covariates(train_data, known_covariate_columns(driver_configs))
    historical_covariates = _available_covariates(
        train_data, historical_covariate_columns(driver_configs)
    )
    for item_id, series in train_data.groupby("item_id", sort=True):
        series = series.sort_values("timestamp").reset_index(drop=True)
        supervised = _make_supervised_frame(
            series, lag_values, known_covariates, historical_covariates
        )
        if supervised.shape[0] < 10:
            continue
        feature_columns = [column for column in supervised.columns if column != "y"]
        model = XGBRegressor(
            n_estimators=160,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=DEFAULT_RANDOM_SEED,
            n_jobs=1,
        )
        model.fit(supervised[feature_columns], supervised["y"])
        future_timestamps = _timestamps_for_item(
            actual_data, item_id, series, freq, prediction_length
        )
        history_values = list(series["target"].astype(float))
        last_historical_values = {
            column: float(series[column].dropna().iloc[-1]) for column in historical_covariates
        }
        known_values = actual_data[actual_data["item_id"].eq(item_id)].copy()
        known_values["timestamp"] = pd.to_datetime(known_values["timestamp"])
        known_values = known_values.set_index("timestamp")
        known_fallbacks = {
            column: float(series[column].dropna().iloc[-1]) for column in known_covariates
        }
        predictions = []
        for timestamp in future_timestamps:
            known_at_timestamp = {
                column: _known_value(
                    known_values, pd.Timestamp(timestamp), column, known_fallbacks[column]
                )
                for column in known_covariates
            }
            features = _feature_row(
                history_values,
                pd.Timestamp(timestamp),
                lag_values,
                known_covariates=known_at_timestamp,
                historical_covariates=last_historical_values,
            )
            prediction = float(model.predict(pd.DataFrame([features]))[0])
            predictions.append(prediction)
            history_values.append(prediction)
        residual_std = float((supervised["y"] - model.predict(supervised[feature_columns])).std())
        interval = 1.28 * residual_std if np.isfinite(residual_std) else 0.0
        frames.append(
            pd.DataFrame(
                {
                    "item_id": item_id,
                    "timestamp": future_timestamps,
                    "forecast_mean": predictions,
                    "forecast_p50": predictions,
                    "forecast_p10": [value - interval for value in predictions],
                    "forecast_p90": [value + interval for value in predictions],
                }
            )
        )
    forecast_data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return _join_actuals(forecast_data, actual_data, "XGBoost", window_id)


def _join_actuals(
    forecast: pd.DataFrame,
    actual_data: pd.DataFrame,
    model: str,
    window_id: str,
) -> pd.DataFrame:
    if forecast.empty:
        return pd.DataFrame(columns=_columns())
    actual = (
        actual_data[["item_id", "timestamp", "target"]] if not actual_data.empty else pd.DataFrame()
    )
    joined = forecast.merge(actual, on=["item_id", "timestamp"], how="left").rename(
        columns={"target": "actual"}
    )
    joined["window_id"] = window_id
    joined["model"] = model
    joined["model_type"] = "外部模型"
    joined["error"] = joined["forecast_p50"] - joined["actual"]
    joined["absolute_error"] = joined["error"].abs()
    joined["error_rate"] = joined.apply(
        lambda row: (
            row["error"] / abs(row["actual"])
            if pd.notna(row["actual"]) and row["actual"] != 0
            else pd.NA
        ),
        axis=1,
    )
    return joined.loc[:, _columns()]


def _future_frame(
    data: pd.DataFrame,
    freq: str,
    prediction_length: int,
    driver_configs: Iterable[ForecastDriverConfig | Mapping[str, object]] | None,
) -> pd.DataFrame:
    future = build_future_known_covariates(
        data, driver_configs, freq=freq, prediction_length=prediction_length
    )
    if future.empty:
        frames = []
        for item_id, series in data.groupby("item_id", sort=True):
            timestamps = pd.date_range(
                pd.to_datetime(series["timestamp"]).max(),
                periods=prediction_length + 1,
                freq=_pandas_freq(freq),
            )[1:]
            frames.append(pd.DataFrame({"item_id": item_id, "timestamp": timestamps}))
        future = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    future["target"] = pd.NA
    return future


def _make_supervised_frame(
    series: pd.DataFrame,
    lag_values: list[int],
    known_covariates: list[str],
    historical_covariates: list[str],
) -> pd.DataFrame:
    values = list(series["target"].astype(float))
    rows = []
    for index in range(max(lag_values), len(series)):
        row = _feature_row(
            values[:index],
            pd.Timestamp(series["timestamp"].iloc[index]),
            lag_values,
            known_covariates={
                column: float(series[column].iloc[index]) for column in known_covariates
            },
            historical_covariates={
                column: float(series[column].iloc[index - 1]) for column in historical_covariates
            },
        )
        row["y"] = values[index]
        rows.append(row)
    return pd.DataFrame(rows)


def _feature_row(
    history_values: list[float],
    timestamp: pd.Timestamp,
    lag_values: list[int],
    *,
    known_covariates: dict[str, float] | None = None,
    historical_covariates: dict[str, float] | None = None,
) -> dict[str, float]:
    row: dict[str, float] = {
        "dayofweek": float(timestamp.dayofweek),
        "month": float(timestamp.month),
        "quarter": float(timestamp.quarter),
        "year_index": float(timestamp.year - 2000),
    }
    for lag in lag_values:
        row[f"lag_{lag}"] = (
            float(history_values[-lag]) if len(history_values) >= lag else float(history_values[-1])
        )
    for window in (3, 7, 14, 30):
        actual_window = min(window, len(history_values))
        row[f"rolling_mean_{window}"] = float(np.mean(history_values[-actual_window:]))
    for column, value in (known_covariates or {}).items():
        row[f"known_{column}"] = value
    for column, value in (historical_covariates or {}).items():
        row[f"historical_{column}_lag_1"] = value
    return row


def _available_covariates(data: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in data and data[column].notna().all()]


def _known_value(
    values: pd.DataFrame,
    timestamp: pd.Timestamp,
    column: str,
    fallback: float,
) -> float:
    if timestamp not in values.index or column not in values:
        return fallback
    value = values.loc[timestamp, column]
    if isinstance(value, pd.Series):
        value = value.iloc[0]
    return float(value) if pd.notna(value) else fallback


def _timestamps_for_item(
    actual_data: pd.DataFrame,
    item_id: str,
    series: pd.DataFrame,
    freq: str,
    prediction_length: int,
) -> pd.DatetimeIndex | pd.Series:
    if not actual_data.empty:
        timestamps = actual_data[actual_data["item_id"].eq(item_id)]["timestamp"]
        if not timestamps.empty:
            return pd.to_datetime(timestamps).sort_values()
    return pd.date_range(
        pd.to_datetime(series["timestamp"]).max(),
        periods=prediction_length + 1,
        freq=_pandas_freq(freq),
    )[1:]


def _statsforecast_freq(freq: str) -> str:
    return {"M": "MS", "W": "W-MON", "D": "D"}[freq]


def _pandas_freq(freq: str) -> str:
    return {"M": "MS", "W": "W-MON", "D": "D"}[freq]


def _season_length(freq: str) -> int:
    return SEASONAL_LAG[freq]


def _xgboost_lags(freq: str) -> list[int]:
    if freq == "D":
        return [1, 2, 3, 7, 14, 30]
    if freq == "W":
        return [1, 2, 4, 8, 13, 26]
    return [1, 2, 3, 6, 12]


def _columns() -> list[str]:
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
