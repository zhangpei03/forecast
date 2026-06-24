from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.domain.models import ExperimentConfig
from src.services.forecast_driver_service import (
    build_future_known_covariates,
    known_covariate_columns,
)


class AutoGluonUnavailableError(RuntimeError):
    pass


def generate_autogluon_backtest_predictions(
    *,
    data: pd.DataFrame,
    config: ExperimentConfig,
    model_dir: Path,
) -> pd.DataFrame:
    try:
        from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
    except ImportError as exc:
        raise AutoGluonUnavailableError(
            "AutoGluon TimeSeries 未安装或当前 Python 环境不兼容。"
        ) from exc

    frames: list[pd.DataFrame] = []
    known_covariates = known_covariate_columns(config.driver_configs)
    clean_data = data.dropna(subset=["item_id", "timestamp", "target"]).sort_values(
        ["item_id", "timestamp"]
    )
    for window_index in range(config.num_val_windows):
        train_frames = []
        actual_frames = []
        for _, series in clean_data.groupby("item_id"):
            test_end = len(series) - window_index * config.prediction_length
            test_start = test_end - config.prediction_length
            if test_start <= 0:
                continue
            train_frames.append(series.iloc[:test_start])
            actual_frames.append(series.iloc[test_start:test_end])
        if not train_frames:
            continue
        train_df = pd.concat(train_frames, ignore_index=True)
        actual_df = pd.concat(actual_frames, ignore_index=True)
        train_ts = TimeSeriesDataFrame.from_data_frame(
            train_df[["item_id", "timestamp", "target", *known_covariates]],
            id_column="item_id",
            timestamp_column="timestamp",
        )
        predictor = TimeSeriesPredictor(
            target="target",
            known_covariates_names=known_covariates,
            prediction_length=config.prediction_length,
            freq=config.freq,
            eval_metric="WAPE",
            quantile_levels=config.quantile_levels,
            path=str(model_dir / f"window_{window_index + 1}"),
        )
        predictor.fit(
            train_ts,
            presets=config.preset,
            hyperparameters=_lightweight_hyperparameters(),
            time_limit=max(120, int(config.time_limit_seconds / max(config.num_val_windows, 1))),
            random_seed=config.random_seed,
            enable_ensemble=True,
        )
        future_covariates = None
        if known_covariates:
            future_index = predictor.make_future_data_frame(train_ts).reset_index()
            future_covariates = TimeSeriesDataFrame.from_data_frame(
                _align_known_covariates_to_future(future_index, actual_df, known_covariates),
                id_column="item_id",
                timestamp_column="timestamp",
            )
        raw_predictions = predictor.predict(train_ts, known_covariates=future_covariates)
        frames.append(
            _format_autogluon_predictions(raw_predictions, actual_df, f"W{window_index + 1}")
        )

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def generate_autogluon_future_forecast(
    *,
    data: pd.DataFrame,
    config: ExperimentConfig,
    model_dir: Path,
    model_name: str,
) -> pd.DataFrame:
    try:
        from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
    except ImportError as exc:
        raise AutoGluonUnavailableError(
            "AutoGluon TimeSeries 未安装或当前 Python 环境不兼容。"
        ) from exc

    known_covariates = known_covariate_columns(config.driver_configs)
    train_ts = TimeSeriesDataFrame.from_data_frame(
        data[["item_id", "timestamp", "target", *known_covariates]],
        id_column="item_id",
        timestamp_column="timestamp",
    )
    predictor = TimeSeriesPredictor(
        target="target",
        known_covariates_names=known_covariates,
        prediction_length=config.prediction_length,
        freq=config.freq,
        eval_metric="WAPE",
        quantile_levels=config.quantile_levels,
        path=str(model_dir / "full"),
    )
    predictor.fit(
        train_ts,
        presets=config.preset,
        hyperparameters=_lightweight_hyperparameters(),
        time_limit=config.time_limit_seconds,
        random_seed=config.random_seed,
        enable_ensemble=True,
    )
    model_for_prediction = None if model_name in {"", "AutoGluon"} else model_name
    future_covariates = None
    if known_covariates:
        future_data = build_future_known_covariates(
            data,
            config.driver_configs,
            freq=config.freq,
            prediction_length=config.prediction_length,
        )
        future_index = predictor.make_future_data_frame(train_ts).reset_index()
        future_covariates = TimeSeriesDataFrame.from_data_frame(
            _align_known_covariates_to_future(future_index, future_data, known_covariates),
            id_column="item_id",
            timestamp_column="timestamp",
        )
    raw_predictions = predictor.predict(
        train_ts,
        known_covariates=future_covariates,
        model=model_for_prediction,
    )
    future = raw_predictions.reset_index()
    future = future.rename(
        columns={
            "mean": "forecast_mean",
            "0.1": "forecast_p10",
            "0.5": "forecast_p50",
            "0.9": "forecast_p90",
        }
    )
    future["model"] = model_name or "AutoGluon"
    future["model_type"] = "AutoGluon"
    future["actual"] = pd.NA
    future["error"] = pd.NA
    future["absolute_error"] = pd.NA
    future["error_rate"] = pd.NA
    future["window_id"] = "FUTURE"
    return future[
        [
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
    ]


def _format_autogluon_predictions(
    raw_predictions: pd.DataFrame,
    actual: pd.DataFrame,
    window_id: str,
) -> pd.DataFrame:
    predictions = raw_predictions.reset_index().rename(
        columns={
            "mean": "forecast_mean",
            "0.1": "forecast_p10",
            "0.5": "forecast_p50",
            "0.9": "forecast_p90",
        }
    )
    if "forecast_p50" not in predictions.columns:
        predictions["forecast_p50"] = predictions["forecast_mean"]
    aligned_actual = _align_known_covariates_to_future(
        predictions[["item_id", "timestamp"]], actual, ["target"]
    )
    joined = predictions.merge(
        aligned_actual,
        on=["item_id", "timestamp"],
        how="left",
    ).rename(columns={"target": "actual"})
    joined["window_id"] = window_id
    joined["model"] = "AutoGluon"
    joined["model_type"] = "AutoGluon"
    joined["error"] = joined["forecast_p50"] - joined["actual"]
    joined["absolute_error"] = joined["error"].abs()
    joined["error_rate"] = joined.apply(
        lambda row: row["error"] / abs(row["actual"]) if row["actual"] != 0 else pd.NA,
        axis=1,
    )
    return joined[
        [
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
    ]


def _lightweight_hyperparameters() -> dict[str, dict]:
    return {
        "SeasonalNaive": {},
        "RecursiveTabular": {},
        "DirectTabular": {},
        "ETS": {},
        "Theta": {},
    }


def _align_known_covariates_to_future(
    future_index: pd.DataFrame,
    source: pd.DataFrame,
    known_covariates: list[str],
) -> pd.DataFrame:
    """Map covariate values by per-series horizon order, not timestamp representation."""

    frames: list[pd.DataFrame] = []
    for item_id, future_item in future_index.groupby("item_id", sort=False):
        future_item = future_item.sort_values("timestamp").copy()
        source_item = source[source["item_id"].eq(item_id)].sort_values("timestamp")
        if len(source_item) < len(future_item):
            raise ValueError(f"协变量缺少预测期数据：{item_id}")
        values = (
            source_item.loc[:, known_covariates].iloc[: len(future_item)].reset_index(drop=True)
        )
        future_item.loc[:, known_covariates] = values.to_numpy()
        frames.append(future_item)
    return pd.concat(frames, ignore_index=True) if frames else future_index.copy()
