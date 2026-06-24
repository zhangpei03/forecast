from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict

import numpy as np
import pandas as pd

from src.domain.models import ForecastDriverConfig

CONFIG_TYPE_COVARIATE = "covariate"
CONFIG_TYPE_GROWTH_RATE = "growth_rate"
CONFIG_TYPE_CALENDAR_FACTOR = "calendar_factor"
CONFIG_TYPE_SCENARIO_COVARIATE = "scenario_covariate"
AVAILABILITY_KNOWN_FUTURE = "known_future"
AVAILABILITY_HISTORICAL = "historical"
FUTURE_VALUE_LAST = "last_value"
FUTURE_VALUE_MEAN = "mean_value"


def normalize_driver_configs(
    configs: Iterable[ForecastDriverConfig | Mapping[str, object]] | None,
) -> list[ForecastDriverConfig]:
    normalized: list[ForecastDriverConfig] = []
    for config in configs or []:
        if isinstance(config, ForecastDriverConfig):
            normalized.append(config)
        else:
            normalized.append(ForecastDriverConfig(**dict(config)))
    return normalized


def driver_configs_to_dicts(
    configs: Iterable[ForecastDriverConfig | Mapping[str, object]] | None,
) -> list[dict[str, object]]:
    return [asdict(config) for config in normalize_driver_configs(configs)]


def known_covariate_columns(
    configs: Iterable[ForecastDriverConfig | Mapping[str, object]] | None,
) -> list[str]:
    return _covariate_columns(configs, availability=AVAILABILITY_KNOWN_FUTURE)


def historical_covariate_columns(
    configs: Iterable[ForecastDriverConfig | Mapping[str, object]] | None,
) -> list[str]:
    return _covariate_columns(configs, availability=AVAILABILITY_HISTORICAL)


def validate_driver_configs(
    configs: Iterable[ForecastDriverConfig | Mapping[str, object]] | None,
    data: pd.DataFrame,
) -> list[str]:
    errors: list[str] = []
    names: set[str] = set()
    columns: set[str] = set()
    for config in normalize_driver_configs(configs):
        name = config.name.strip()
        if not name:
            errors.append("预测驱动配置名称不能为空。")
            continue
        if name in names:
            errors.append(f"预测驱动配置名称重复：{name}。")
        names.add(name)
        if config.config_type == CONFIG_TYPE_COVARIATE:
            if not config.column:
                errors.append(f"协变量配置“{name}”未选择字段。")
                continue
            if config.column not in data.columns:
                errors.append(f"协变量配置“{name}”字段不存在：{config.column}。")
                continue
            if config.column in columns:
                errors.append(f"协变量字段重复配置：{config.column}。")
            columns.add(config.column)
            numeric_values = pd.to_numeric(data[config.column], errors="coerce")
            if numeric_values.notna().sum() != len(data):
                errors.append(f"协变量配置“{name}”必须使用无缺失的数值字段。")
            if config.availability not in {AVAILABILITY_KNOWN_FUTURE, AVAILABILITY_HISTORICAL}:
                errors.append(f"协变量配置“{name}”缺少可用性类型。")
            if (
                config.availability == AVAILABILITY_KNOWN_FUTURE
                and config.future_value_strategy not in {FUTURE_VALUE_LAST, FUTURE_VALUE_MEAN}
            ):
                errors.append(f"协变量配置“{name}”缺少未来值生成规则。")
        elif config.config_type == CONFIG_TYPE_GROWTH_RATE:
            if config.growth_rate is None or not np.isfinite(float(config.growth_rate)):
                errors.append(f"增长率配置“{name}”必须填写有效数值。")
            elif not -1 < float(config.growth_rate) <= 10:
                errors.append(f"增长率配置“{name}”必须大于 -100% 且不超过 1000%。")
        elif config.config_type == CONFIG_TYPE_CALENDAR_FACTOR:
            if not config.impact_months or any(
                month < 1 or month > 12 for month in config.impact_months
            ):
                errors.append(f"事件影响配置“{name}”必须选择 1 至 12 月的影响月份。")
            if config.impact_rate is None or not np.isfinite(float(config.impact_rate)):
                errors.append(f"事件影响配置“{name}”必须填写影响率。")
            elif not -1 < float(config.impact_rate) <= 10:
                errors.append(f"事件影响配置“{name}”影响率必须大于 -100% 且不超过 1000%。")
        elif config.config_type == CONFIG_TYPE_SCENARIO_COVARIATE:
            if config.base_value is None or not np.isfinite(float(config.base_value)):
                errors.append(f"手工情景协变量“{name}”必须填写基准值。")
            elif float(config.base_value) == 0:
                errors.append(f"手工情景协变量“{name}”基准值不能为 0。")
            for field_name, value in {
                "每期变化率": config.scenario_growth_rate,
                "目标影响系数": config.effect_rate,
            }.items():
                if value is None or not np.isfinite(float(value)):
                    errors.append(f"手工情景协变量“{name}”必须填写{field_name}。")
        else:
            errors.append(f"预测驱动配置“{name}”类型无效。")
    return errors


def build_future_known_covariates(
    data: pd.DataFrame,
    configs: Iterable[ForecastDriverConfig | Mapping[str, object]] | None,
    *,
    freq: str,
    prediction_length: int,
) -> pd.DataFrame:
    configs_by_column = {
        config.column: config
        for config in normalize_driver_configs(configs)
        if config.enabled
        and config.config_type == CONFIG_TYPE_COVARIATE
        and config.availability == AVAILABILITY_KNOWN_FUTURE
        and config.column
    }
    if not configs_by_column:
        return pd.DataFrame(columns=["item_id", "timestamp"])

    frames: list[pd.DataFrame] = []
    for item_id, series in data.groupby("item_id", sort=True):
        ordered = series.sort_values("timestamp")
        timestamps = pd.date_range(
            pd.to_datetime(ordered["timestamp"]).max(),
            periods=prediction_length + 1,
            freq=_pandas_freq(freq),
        )[1:]
        frame = pd.DataFrame({"item_id": item_id, "timestamp": timestamps})
        for column, config in configs_by_column.items():
            values = pd.to_numeric(ordered[column], errors="coerce").dropna()
            if config.future_value_strategy == FUTURE_VALUE_MEAN:
                value = float(values.mean()) if not values.empty else np.nan
            else:
                value = float(values.iloc[-1]) if not values.empty else np.nan
            frame[column] = value
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def apply_growth_rate_adjustments(
    predictions: pd.DataFrame,
    configs: Iterable[ForecastDriverConfig | Mapping[str, object]] | None,
) -> pd.DataFrame:
    rates = [
        float(config.growth_rate)
        for config in normalize_driver_configs(configs)
        if config.enabled
        and config.config_type == CONFIG_TYPE_GROWTH_RATE
        and config.growth_rate is not None
    ]
    if predictions.empty or not rates:
        return predictions.copy()

    adjusted = predictions.copy()
    period_multiplier = float(np.prod([1 + rate for rate in rates]))
    group_columns = [column for column in ["model", "item_id", "window_id"] if column in adjusted]
    ordered = adjusted.sort_values([*group_columns, "timestamp"]).copy()
    ordered["_growth_factor"] = (
        ordered.groupby(group_columns, sort=False)
        .cumcount()
        .add(1)
        .map(lambda horizon: period_multiplier**horizon)
    )
    forecast_columns = [
        column
        for column in ["forecast_mean", "forecast_p10", "forecast_p50", "forecast_p90"]
        if column in ordered
    ]
    for column in forecast_columns:
        ordered[column] = (
            pd.to_numeric(ordered[column], errors="coerce") * ordered["_growth_factor"]
        )
    if {"actual", "forecast_p50"}.issubset(ordered.columns):
        ordered["error"] = ordered["forecast_p50"] - ordered["actual"]
        ordered["absolute_error"] = ordered["error"].abs()
        ordered["error_rate"] = ordered.apply(
            lambda row: (
                row["error"] / abs(row["actual"])
                if pd.notna(row["actual"]) and row["actual"] != 0
                else pd.NA
            ),
            axis=1,
        )
    return ordered.drop(columns="_growth_factor").sort_index()


def apply_forecast_driver_adjustments(
    predictions: pd.DataFrame,
    configs: Iterable[ForecastDriverConfig | Mapping[str, object]] | None,
) -> pd.DataFrame:
    adjusted = apply_growth_rate_adjustments(predictions, configs)
    parsed_configs = normalize_driver_configs(configs)
    for config in parsed_configs:
        if not config.enabled:
            continue
        if config.config_type == CONFIG_TYPE_CALENDAR_FACTOR and config.impact_rate is not None:
            months = set(config.impact_months)
            multiplier = pd.Series(
                np.where(
                    pd.to_datetime(adjusted["timestamp"]).dt.month.isin(months),
                    1 + float(config.impact_rate),
                    1.0,
                ),
                index=adjusted.index,
            )
            adjusted = _apply_multiplier(adjusted, multiplier)
        if (
            config.config_type == CONFIG_TYPE_SCENARIO_COVARIATE
            and config.scenario_growth_rate is not None
            and config.effect_rate is not None
        ):
            multiplier = _scenario_multiplier(
                adjusted,
                scenario_growth_rate=float(config.scenario_growth_rate),
                effect_rate=float(config.effect_rate),
            )
            adjusted = _apply_multiplier(adjusted, multiplier)
    return _recalculate_errors(adjusted)


def build_future_driver_assumptions(
    data: pd.DataFrame,
    configs: Iterable[ForecastDriverConfig | Mapping[str, object]] | None,
    *,
    freq: str,
    prediction_length: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    scenario_configs = [
        config
        for config in normalize_driver_configs(configs)
        if config.enabled and config.config_type == CONFIG_TYPE_SCENARIO_COVARIATE
    ]
    for config in scenario_configs:
        if config.base_value is None or config.scenario_growth_rate is None:
            continue
        for item_id, series in data.groupby("item_id", sort=True):
            timestamps = pd.date_range(
                pd.to_datetime(series["timestamp"]).max(),
                periods=prediction_length + 1,
                freq=_pandas_freq(freq),
            )[1:]
            values = float(config.base_value) * (
                1 + float(config.scenario_growth_rate)
            ) ** np.arange(1, prediction_length + 1)
            frames.append(
                pd.DataFrame(
                    {
                        "item_id": item_id,
                        "timestamp": timestamps,
                        "driver_name": config.name,
                        "driver_type": "手工情景协变量",
                        "assumption_value": values,
                        "base_value": float(config.base_value),
                        "period_growth_rate": float(config.scenario_growth_rate),
                        "effect_rate": float(config.effect_rate or 0),
                        "adjustment_mode": "手工影响系数",
                    }
                )
            )
    return (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(
            columns=[
                "item_id",
                "timestamp",
                "driver_name",
                "driver_type",
                "assumption_value",
                "base_value",
                "period_growth_rate",
                "effect_rate",
                "adjustment_mode",
            ]
        )
    )


def _covariate_columns(
    configs: Iterable[ForecastDriverConfig | Mapping[str, object]] | None,
    *,
    availability: str,
) -> list[str]:
    return [
        config.column
        for config in normalize_driver_configs(configs)
        if config.enabled
        and config.config_type == CONFIG_TYPE_COVARIATE
        and config.availability == availability
        and config.column
    ]


def _pandas_freq(freq: str) -> str:
    return {"M": "MS", "W": "W-MON", "D": "D"}[freq]


def _apply_multiplier(predictions: pd.DataFrame, multiplier: pd.Series) -> pd.DataFrame:
    adjusted = predictions.copy()
    for column in ["forecast_mean", "forecast_p10", "forecast_p50", "forecast_p90"]:
        if column in adjusted:
            adjusted[column] = pd.to_numeric(adjusted[column], errors="coerce") * multiplier
    return adjusted


def _scenario_multiplier(
    predictions: pd.DataFrame,
    *,
    scenario_growth_rate: float,
    effect_rate: float,
) -> pd.Series:
    group_columns = [
        column for column in ["model", "item_id", "window_id"] if column in predictions
    ]
    ordered = predictions.sort_values([*group_columns, "timestamp"])
    horizon = ordered.groupby(group_columns, sort=False).cumcount().add(1)
    relative_change = (1 + scenario_growth_rate) ** horizon - 1
    multiplier = 1 + effect_rate * relative_change
    return multiplier.reindex(predictions.index)


def _recalculate_errors(predictions: pd.DataFrame) -> pd.DataFrame:
    adjusted = predictions.copy()
    if {"actual", "forecast_p50"}.issubset(adjusted.columns):
        adjusted["error"] = adjusted["forecast_p50"] - adjusted["actual"]
        adjusted["absolute_error"] = adjusted["error"].abs()
        adjusted["error_rate"] = adjusted.apply(
            lambda row: (
                row["error"] / abs(row["actual"])
                if pd.notna(row["actual"]) and row["actual"] != 0
                else pd.NA
            ),
            axis=1,
        )
    return adjusted
