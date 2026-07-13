from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.core.constants import AUTOGLUON_MODEL_NAMES
from src.domain.models import ExperimentConfig
from src.services.forecast_driver_service import (
    build_future_known_covariates,
    historical_covariate_columns,
    known_covariate_columns,
)


class AutoGluonUnavailableError(RuntimeError):
    pass


def _configure_parallelism(num_workers: int | None) -> int:
    """Set PyTorch threads and return effective worker count for this machine."""
    try:
        import torch
    except ImportError:
        torch = None
    cpu_count = os.cpu_count() or 4
    if num_workers is None:
        num_workers = min(cpu_count, 4)
    if torch is not None:
        torch.set_num_threads(num_workers)
    os.environ.setdefault("OMP_NUM_THREADS", str(num_workers))
    return num_workers


def generate_autogluon_backtest_predictions(
    *,
    data: pd.DataFrame,
    config: ExperimentConfig,
    model_dir: Path,
    selected_model: str | None = None,
    selected_models: list[str] | None = None,
) -> pd.DataFrame:
    selected_models = _normalize_model_names(
        selected_models if selected_models is not None else ([selected_model] if selected_model else None)
    )
    num_workers = _configure_parallelism(config.num_workers)
    try:
        from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
    except ImportError as exc:
        raise AutoGluonUnavailableError(
            "AutoGluon TimeSeries 未安装或当前 Python 环境不兼容。"
        ) from exc

    frames: list[pd.DataFrame] = []
    clean_data = data.dropna(subset=["item_id", "timestamp", "target"]).sort_values(
        ["item_id", "timestamp"]
    )
    known_covariates, past_covariates = _autogluon_covariate_columns(config, clean_data)
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
            train_df[_time_series_input_columns(train_df, known_covariates, past_covariates)],
            id_column="item_id",
            timestamp_column="timestamp",
            static_features_df=_static_features_frame(train_df, config.static_features),
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
            hyperparameters=hyperparameters_for_preset(config.preset, selected_models),
            time_limit=max(120, int(config.time_limit_seconds / max(config.num_val_windows, 1))),
            random_seed=config.random_seed,
            enable_ensemble=selected_models is None,
            num_workers=num_workers,
        )
        future_covariates = None
        if known_covariates:
            future_index = predictor.make_future_data_frame(train_ts).reset_index()
            future_covariates = TimeSeriesDataFrame.from_data_frame(
                _align_known_covariates_to_future(future_index, actual_df, known_covariates),
                id_column="item_id",
                timestamp_column="timestamp",
            )
        for internal_model_name, output_model_name in _prediction_model_pairs(
            predictor,
            selected_models,
        ):
            predict_kwargs = {"known_covariates": future_covariates}
            if internal_model_name is not None:
                predict_kwargs["model"] = internal_model_name
            predict_kwargs["num_workers"] = num_workers
            raw_predictions = predictor.predict(train_ts, **predict_kwargs)
            frames.append(
                _format_autogluon_predictions(
                    raw_predictions,
                    actual_df,
                    f"W{window_index + 1}",
                    model_name=output_model_name,
                )
            )

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def generate_autogluon_future_forecast(
    *,
    data: pd.DataFrame,
    config: ExperimentConfig,
    model_dir: Path,
    model_name: str,
    selected_model: str | None = None,
) -> pd.DataFrame:
    selected_model = _normalize_model_name(selected_model)
    model_name = _normalize_model_name(model_name) or ""
    num_workers = _configure_parallelism(config.num_workers)
    try:
        from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
    except ImportError as exc:
        raise AutoGluonUnavailableError(
            "AutoGluon TimeSeries 未安装或当前 Python 环境不兼容。"
        ) from exc

    training_data = data.dropna(subset=["item_id", "timestamp", "target"]).copy()
    known_covariates, past_covariates = _autogluon_covariate_columns(config, training_data)
    train_ts = TimeSeriesDataFrame.from_data_frame(
        training_data[
            _time_series_input_columns(training_data, known_covariates, past_covariates)
        ],
        id_column="item_id",
        timestamp_column="timestamp",
        static_features_df=_static_features_frame(training_data, config.static_features),
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
        hyperparameters=hyperparameters_for_preset(config.preset, selected_model),
        time_limit=config.time_limit_seconds,
        random_seed=config.random_seed,
        enable_ensemble=selected_model is None,
        num_workers=num_workers,
    )
    model_for_prediction = None if selected_model or model_name in {"", "AutoGluon"} else model_name
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
        num_workers=num_workers,
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
    future["model"] = selected_model or model_name or "AutoGluon"
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
    *,
    model_name: str = "AutoGluon",
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
    joined["model"] = model_name
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
    """快速验证 / 标准评测档位: 统计 + 树模型, CPU 上即可快速收敛。"""

    return {
        "SeasonalNaive": {},
        "RecursiveTabular": {},
        "DirectTabular": {},
        "ETS": {},
        "Theta": {},
    }


def _standard_hyperparameters() -> dict[str, dict]:
    """标准评测档位: 在轻量集基础上补 LightGBM 后端的树模型。"""

    hyperparameters = _lightweight_hyperparameters()
    hyperparameters["RecursiveTabular"] = {"model_name": "GBM"}
    hyperparameters["DirectTabular"] = {"model_name": "GBM"}
    return hyperparameters


def _deep_hyperparameters() -> dict[str, dict]:
    """深度评测档位: 在标准集基础上放开深度学习与预训练时序模型。"""

    hyperparameters = _standard_hyperparameters()
    hyperparameters["DeepAR"] = {}
    hyperparameters["TemporalFusionTransformer"] = {}
    hyperparameters["PatchTST"] = {}
    hyperparameters["Chronos2"] = {"model_path": "autogluon/chronos-2"}
    return hyperparameters


def hyperparameters_for_preset(
    preset: str,
    selected_models: str | list[str] | None = None,
) -> dict[str, dict]:
    """按训练档位选择 AutoGluon 候选模型集合。

    - ``fast_training``: 轻量统计 + 树模型, 最快给出可预测性结论。
    - ``medium_quality``: 在轻量集基础上补 LightGBM 后端。
    - ``high_quality``: 放开 DeepAR / TFT / PatchTST / Chronos 深度模型。
    """

    if preset == "high_quality":
        hyperparameters = _deep_hyperparameters()
    elif preset == "fast_training":
        hyperparameters = _lightweight_hyperparameters()
    else:
        hyperparameters = _standard_hyperparameters()
    selected_model_names = _normalize_model_names(selected_models)
    if not selected_model_names:
        return hyperparameters
    unsupported = [
        model_name for model_name in selected_model_names if model_name not in AUTOGLUON_MODEL_NAMES
    ]
    if unsupported:
        raise ValueError(f"Unsupported AutoGluon model: {', '.join(unsupported)}")
    return {
        model_name: hyperparameters.get(model_name, _single_model_defaults(model_name))
        for model_name in selected_model_names
    }


def _normalize_model_name(model_name: str | None) -> str | None:
    if model_name == "Chronos":
        return "Chronos2"
    return model_name


def _normalize_model_names(model_names: str | list[str] | None) -> list[str] | None:
    if model_names is None:
        return None
    if isinstance(model_names, str):
        model_names = [model_names]
    normalized = []
    for model_name in model_names:
        clean_name = _normalize_model_name(str(model_name).strip())
        if clean_name and clean_name not in normalized:
            normalized.append(clean_name)
    return normalized or None


def _prediction_model_pairs(
    predictor,
    selected_models: list[str] | None,
) -> list[tuple[str | None, str]]:
    if not selected_models:
        return [(None, "AutoGluon")]
    available = _available_model_names(predictor)
    pairs: list[tuple[str | None, str]] = []
    for selected_model in selected_models:
        internal_model = _match_autogluon_model_name(selected_model, available)
        pairs.append((internal_model, selected_model))
    return pairs


def _available_model_names(predictor) -> list[str]:
    try:
        return list(predictor.model_names())
    except Exception:
        return []


def _match_autogluon_model_name(selected_model: str, available_models: list[str]) -> str | None:
    if selected_model in available_models:
        return selected_model
    for model_name in available_models:
        if model_name.startswith(selected_model):
            return model_name
    if selected_model == "Chronos2":
        for model_name in available_models:
            if model_name.startswith("Chronos"):
                return model_name
    return selected_model


def _single_model_defaults(model_name: str) -> dict:
    if model_name == "Chronos2":
        return {"model_path": "autogluon/chronos-2"}
    if model_name in {"RecursiveTabular", "DirectTabular"}:
        return {"model_name": "GBM"}
    return {}


def _autogluon_covariate_columns(
    config: ExperimentConfig,
    data: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    known_candidates = [
        *config.known_covariates,
        *known_covariate_columns(config.driver_configs),
    ]
    known_covariates = _available_feature_columns(data, known_candidates)
    past_candidates = [
        *config.past_covariates,
        *historical_covariate_columns(config.driver_configs),
    ]
    past_covariates = _available_feature_columns(
        data,
        past_candidates,
        exclude=known_covariates,
    )
    return known_covariates, past_covariates


def _time_series_input_columns(
    data: pd.DataFrame,
    known_covariates: list[str],
    past_covariates: list[str],
) -> list[str]:
    return [
        column
        for column in dict.fromkeys(
            ["item_id", "timestamp", "target", *known_covariates, *past_covariates]
        )
        if column in data.columns
    ]


def _static_features_frame(
    data: pd.DataFrame,
    static_features: list[str],
) -> pd.DataFrame | None:
    feature_columns = _available_feature_columns(
        data,
        static_features,
        exclude=["item_id", "timestamp", "target"],
    )
    if not feature_columns:
        return None
    static_frame = (
        data.sort_values(["item_id", "timestamp"])
        .loc[:, ["item_id", *feature_columns]]
        .dropna(subset=["item_id"])
        .groupby("item_id", sort=False, as_index=False)
        .first()
        .set_index("item_id")
    )
    return static_frame if not static_frame.empty else None


def _available_feature_columns(
    data: pd.DataFrame,
    columns: list[str],
    *,
    exclude: list[str] | None = None,
) -> list[str]:
    excluded = set(exclude or [])
    return [
        column
        for column in dict.fromkeys(columns)
        if column and column in data.columns and column not in excluded
    ]


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
