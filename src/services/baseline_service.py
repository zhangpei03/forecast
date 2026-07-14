from __future__ import annotations

import math
import re

import pandas as pd

from src.core.constants import (
    BASELINE_MODEL_NAMES,
    PREDICTION_PROVENANCE_COLUMNS,
    ROLLING_MEAN_WINDOW,
    SEASONAL_LAG,
)

YOY_WEEKDAY_WOW_MODEL = "YoY Weekday WoW"
YOY_WEEKDAY_DOD_MODEL = "YoY Weekday DoD (Holiday)"
DOD2_MODEL = "YoY Weekday DoD (Weekday Only)"
YOY_WEEKDAY_DOD_ANCHOR_MODEL = "YoY Weekday DoD (Anchor v1)"
YOY_WEEKDAY_DOD_ANCHOR_V2_MODEL = "YoY Weekday DoD (Anchor v2)"
LEGACY_DOD_MODEL_ALIASES = {
    "YoY Weekday DoD": YOY_WEEKDAY_DOD_MODEL,
    "DoD2": DOD2_MODEL,
    "YoY Weekday DoD Anchor": YOY_WEEKDAY_DOD_ANCHOR_MODEL,
    "YoY Weekday DoD Anchor v2": YOY_WEEKDAY_DOD_ANCHOR_V2_MODEL,
}
BASELINE_MODELS = BASELINE_MODEL_NAMES

YOY_LAG = {"M": 12, "W": 52, "D": 365}
WOW_LAG = {"M": 1, "W": 1, "D": 7}
YOY_RATIO_SEARCH_WEEKS = 4
YOY_LEVEL_SCALE_LOOKBACK_DAYS = 28
DOD_ANCHOR_HOLIDAY_RECURSIVE_WEIGHT = 0.60
DOD_ANCHOR_POST_HOLIDAY_RECURSIVE_WEIGHT = 0.70
DOD_ANCHOR_WEEKLY_RECURSIVE_WEIGHT = 0.80
DOD_ANCHOR_POST_HOLIDAY_DAYS = 3
ANCHOR_V2_WEIGHT_LOOKBACK_DAYS = 56
ANCHOR_V2_WEIGHT_PRIOR_STRENGTH = 14
ANCHOR_V2_DEFAULT_RECURSIVE_WEIGHTS = {
    "holiday": 0.30,
    "post_holiday": 0.20,
    "normal": 0.50,
}
WEATHER_COLUMN_TOKENS = ("天气", "雨雪", "weather", "rain", "snow")
HOLIDAY_COLUMN_TOKENS = ("特殊假期",)


def generate_baseline_backtest_predictions(
    *,
    data: pd.DataFrame,
    freq: str,
    prediction_length: int,
    num_windows: int,
    models: list[str] | None = None,
) -> pd.DataFrame:
    clean_data = data.dropna(subset=["item_id", "timestamp", "target"]).copy()
    clean_data["timestamp"] = pd.to_datetime(clean_data["timestamp"])
    clean_data = clean_data.sort_values(["item_id", "timestamp"])
    frames: list[pd.DataFrame] = []
    selected_models = tuple(
        _canonical_baseline_model(model) for model in (models or _baseline_models_for_freq(freq))
    )

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
            for model in selected_models:
                frame = _predict_baseline_model(model, item_id, actual, train, freq, window_index)
                if frame is not None:
                    frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=_prediction_columns())
    predictions = pd.concat(frames, ignore_index=True)
    return _with_error_columns(predictions)


def _baseline_models_for_freq(freq: str) -> tuple[str, ...]:
    common = ("Last Value", "Seasonal Naive", "Rolling Mean")
    if freq == "D":
        return (
            *common,
            "YoY",
            "WoW",
            "MTD Daily Avg",
            YOY_WEEKDAY_WOW_MODEL,
            YOY_WEEKDAY_DOD_MODEL,
            YOY_WEEKDAY_DOD_ANCHOR_MODEL,
            YOY_WEEKDAY_DOD_ANCHOR_V2_MODEL,
            DOD2_MODEL,
        )
    return common


def _canonical_baseline_model(model: str) -> str:
    """Accept previous DoD labels while emitting the standardized model name."""
    return LEGACY_DOD_MODEL_ALIASES.get(model, model)


def _predict_baseline_model(
    model: str,
    item_id: str,
    actual: pd.DataFrame,
    train: pd.DataFrame,
    freq: str,
    window_index: int,
) -> pd.DataFrame | None:
    if model == "Last Value":
        return _predict_last_value(item_id, actual, train, window_index)
    if model == "Seasonal Naive":
        return _predict_seasonal_naive(item_id, actual, train, freq, window_index)
    if model == "Rolling Mean":
        return _predict_rolling_mean(item_id, actual, train, freq, window_index)
    if model == "YoY" and freq == "D":
        return _predict_yoy(item_id, actual, train, freq, window_index)
    if model == "WoW" and freq == "D":
        return _predict_wow(item_id, actual, train, freq, window_index)
    if model == "MTD Daily Avg" and freq == "D":
        return _predict_mtd_daily_avg(item_id, actual, train, window_index)
    if model == YOY_WEEKDAY_WOW_MODEL and freq == "D":
        return _predict_yoy_weekday_wow_original(item_id, actual, train, window_index)
    if model == YOY_WEEKDAY_DOD_MODEL and freq == "D":
        return _predict_yoy_weekday_dod(item_id, actual, train, window_index)
    if model == YOY_WEEKDAY_DOD_ANCHOR_MODEL and freq == "D":
        return _predict_yoy_weekday_dod_anchor(item_id, actual, train, window_index)
    if model == YOY_WEEKDAY_DOD_ANCHOR_V2_MODEL and freq == "D":
        return _predict_yoy_weekday_dod_anchor_v2(item_id, actual, train, window_index)
    if model == DOD2_MODEL and freq == "D":
        return _predict_dod2(item_id, actual, train, window_index)
    return None


def generate_baseline_future_forecast(
    *,
    data: pd.DataFrame,
    freq: str,
    prediction_length: int,
    model: str,
) -> pd.DataFrame:
    model = _canonical_baseline_model(model)
    source_data = data.dropna(subset=["item_id", "timestamp"]).copy()
    source_data["timestamp"] = pd.to_datetime(source_data["timestamp"])
    source_data = source_data.sort_values(["item_id", "timestamp"])
    clean_data = source_data.dropna(subset=["target"])
    pandas_freq = {"M": "MS", "W": "W-MON", "D": "D"}[freq]
    frames = []
    for item_id, series in clean_data.groupby("item_id", sort=True):
        series = series.sort_values("timestamp").reset_index(drop=True)
        context = source_data[source_data["item_id"].eq(item_id)].copy()
        future_dates = pd.date_range(
            series["timestamp"].iloc[-1],
            periods=prediction_length + 1,
            freq=pandas_freq,
        )[1:]
        actual = pd.DataFrame({"timestamp": future_dates, "target": [pd.NA] * prediction_length})
        provenance: list[dict[str, object]] | None = None
        if model == "Last Value":
            forecasts = [float(series["target"].iloc[-1])] * prediction_length
        elif model == "Seasonal Naive":
            lag = SEASONAL_LAG[freq]
            fallback = float(series["target"].iloc[-1])
            history = list(series["target"].astype(float))
            forecasts = []
            for horizon in range(prediction_length):
                forecasts.append(_seasonal_value(history, forecasts, lag, horizon, fallback))
        elif model == "YoY":
            lag = YOY_LAG[freq]
            fallback = float(series["target"].iloc[-1])
            history = list(series["target"].astype(float))
            forecasts = []
            for horizon in range(prediction_length):
                forecasts.append(_seasonal_value(history, forecasts, lag, horizon, fallback))
        elif model == "WoW":
            lag = WOW_LAG[freq]
            fallback = float(series["target"].iloc[-1])
            history = list(series["target"].astype(float))
            forecasts = []
            for horizon in range(prediction_length):
                forecasts.append(_seasonal_value(history, forecasts, lag, horizon, fallback))
        elif model == "MTD Daily Avg":
            forecasts = _mtd_forecasts(series, future_dates)
        elif model == YOY_WEEKDAY_WOW_MODEL:
            forecasts, provenance = _yoy_weekday_wow_original_forecasts(
                history=series,
                target_dates=future_dates,
                context=context,
            )
        elif model == YOY_WEEKDAY_DOD_MODEL:
            forecasts, provenance = _yoy_weekday_dod_forecasts(
                history=series,
                target_dates=future_dates,
                context=context,
            )
        elif model == YOY_WEEKDAY_DOD_ANCHOR_MODEL:
            forecasts, provenance = _yoy_weekday_dod_anchor_forecasts(
                history=series,
                target_dates=future_dates,
                context=context,
            )
        elif model == YOY_WEEKDAY_DOD_ANCHOR_V2_MODEL:
            forecasts, provenance = _yoy_weekday_dod_anchor_v2_forecasts(
                history=series,
                target_dates=future_dates,
                context=context,
            )
        elif model == DOD2_MODEL:
            forecasts, provenance = _dod2_forecasts(
                history=series,
                target_dates=future_dates,
            )
        else:
            window = ROLLING_MEAN_WINDOW[freq]
            forecasts = [float(series["target"].tail(window).mean())] * prediction_length
            provenance = None
        frame = _prediction_frame(model, item_id, actual, forecasts, -1, provenance=provenance)
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


def _predict_yoy(
    item_id: str,
    actual: pd.DataFrame,
    train: pd.DataFrame,
    freq: str,
    window_index: int,
) -> pd.DataFrame:
    lag = YOY_LAG[freq]
    fallback = float(train["target"].iloc[-1])
    history = list(train["target"].astype(float))
    forecasts = [_seasonal_value(history, [], lag, h, fallback) for h in range(len(actual))]
    return _prediction_frame("YoY", item_id, actual, forecasts, window_index)


def _predict_wow(
    item_id: str,
    actual: pd.DataFrame,
    train: pd.DataFrame,
    freq: str,
    window_index: int,
) -> pd.DataFrame:
    lag = WOW_LAG[freq]
    fallback = float(train["target"].iloc[-1])
    history = list(train["target"].astype(float))
    forecasts = [_seasonal_value(history, [], lag, h, fallback) for h in range(len(actual))]
    return _prediction_frame("WoW", item_id, actual, forecasts, window_index)


def _predict_mtd_daily_avg(
    item_id: str,
    actual: pd.DataFrame,
    train: pd.DataFrame,
    window_index: int,
) -> pd.DataFrame:
    actual_ts = pd.to_datetime(actual["timestamp"])
    history = train.copy()
    history["timestamp"] = pd.to_datetime(history["timestamp"])
    forecasts = []
    for ts in actual_ts:
        month_start = ts.replace(day=1)
        mtd_history = history[(history["timestamp"] >= month_start) & (history["timestamp"] < ts)]
        if not mtd_history.empty:
            forecasts.append(float(mtd_history["target"].mean()))
        else:
            same_month = history[history["timestamp"].dt.month == ts.month]
            forecasts.append(
                float(same_month["target"].mean())
                if not same_month.empty
                else float(history["target"].iloc[-1])
            )
    return _prediction_frame("MTD Daily Avg", item_id, actual, forecasts, window_index)


def _predict_yoy_weekday_wow_original(
    item_id: str,
    actual: pd.DataFrame,
    train: pd.DataFrame,
    window_index: int,
) -> pd.DataFrame:
    forecasts, provenance = _yoy_weekday_wow_original_forecasts(
        history=train,
        target_dates=pd.DatetimeIndex(pd.to_datetime(actual["timestamp"])),
        context=pd.concat([train, actual], ignore_index=True),
    )
    return _prediction_frame(
        YOY_WEEKDAY_WOW_MODEL,
        item_id,
        actual,
        forecasts,
        window_index,
        provenance=provenance,
    )


def _predict_yoy_weekday_dod(
    item_id: str,
    actual: pd.DataFrame,
    train: pd.DataFrame,
    window_index: int,
) -> pd.DataFrame:
    forecasts, provenance = _yoy_weekday_dod_forecasts(
        history=train,
        target_dates=pd.DatetimeIndex(pd.to_datetime(actual["timestamp"])),
        context=pd.concat([train, actual], ignore_index=True),
    )
    return _prediction_frame(
        YOY_WEEKDAY_DOD_MODEL,
        item_id,
        actual,
        forecasts,
        window_index,
        provenance=provenance,
    )


def _predict_dod2(
    item_id: str,
    actual: pd.DataFrame,
    train: pd.DataFrame,
    window_index: int,
) -> pd.DataFrame:
    forecasts, provenance = _dod2_forecasts(
        history=train,
        target_dates=pd.DatetimeIndex(pd.to_datetime(actual["timestamp"])),
    )
    return _prediction_frame(
        DOD2_MODEL,
        item_id,
        actual,
        forecasts,
        window_index,
        provenance=provenance,
    )


def _predict_yoy_weekday_dod_anchor(
    item_id: str,
    actual: pd.DataFrame,
    train: pd.DataFrame,
    window_index: int,
) -> pd.DataFrame:
    forecasts, provenance = _yoy_weekday_dod_anchor_forecasts(
        history=train,
        target_dates=pd.DatetimeIndex(pd.to_datetime(actual["timestamp"])),
        context=pd.concat([train, actual], ignore_index=True),
    )
    return _prediction_frame(
        YOY_WEEKDAY_DOD_ANCHOR_MODEL,
        item_id,
        actual,
        forecasts,
        window_index,
        provenance=provenance,
    )


def _predict_yoy_weekday_dod_anchor_v2(
    item_id: str,
    actual: pd.DataFrame,
    train: pd.DataFrame,
    window_index: int,
) -> pd.DataFrame:
    forecasts, provenance = _yoy_weekday_dod_anchor_v2_forecasts(
        history=train,
        target_dates=pd.DatetimeIndex(pd.to_datetime(actual["timestamp"])),
        context=pd.concat([train, actual], ignore_index=True),
    )
    return _prediction_frame(
        YOY_WEEKDAY_DOD_ANCHOR_V2_MODEL,
        item_id,
        actual,
        forecasts,
        window_index,
        provenance=provenance,
    )


def _yoy_weekday_wow_original_forecasts(
    *,
    history: pd.DataFrame,
    target_dates: pd.DatetimeIndex,
    context: pd.DataFrame,
) -> tuple[list[float], list[dict[str, object]]]:
    """Original weekday-aligned YoY WoW baseline from record.py."""
    ordered = history.dropna(subset=["timestamp", "target"]).copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"]).dt.normalize()
    ordered = ordered.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    values = {
        timestamp: float(target)
        for timestamp, target in zip(ordered["timestamp"], ordered["target"], strict=False)
    }
    if not values:
        return [0.0] * len(target_dates), [_empty_provenance("前一周值 × 去年周环比") for _ in target_dates]

    context_indexed = _prepare_context(context)
    fallback = float(ordered["target"].iloc[-1])
    forecasts: list[float] = []
    provenance: list[dict[str, object]] = []
    for raw_timestamp in target_dates:
        timestamp = pd.Timestamp(raw_timestamp).normalize()
        base_timestamp = timestamp - pd.Timedelta(days=7)
        base = values.get(base_timestamp)
        if base is None or not math.isfinite(base):
            base = forecasts[-7] if len(forecasts) >= 7 else fallback
        aligned = _align_to_prior_year_weekday(timestamp)
        comparison_timestamp = aligned - pd.Timedelta(days=7)
        raw_ratio = _weekly_ratio(values, aligned)
        ratio = _select_yoy_weekly_ratio_original(
            timestamp=timestamp,
            history_values=values,
            context=context_indexed,
        )
        forecast = float(base * ratio)
        values[timestamp] = forecast
        forecasts.append(forecast)
        provenance.append(
            _ratio_provenance(
                basis="前一周值 × 去年周环比",
                base_timestamp=base_timestamp,
                base_value=base,
                prior_year_timestamp=aligned,
                prior_year_value=values.get(aligned),
                comparison_timestamp=comparison_timestamp,
                comparison_value=values.get(comparison_timestamp),
                applied_ratio=ratio - 1.0,
                raw_ratio=raw_ratio - 1.0 if raw_ratio is not None else None,
                ratio_name="去年对齐日周环比",
            )
        )
    return forecasts, provenance


def _yoy_weekday_dod_forecasts(
    *,
    history: pd.DataFrame,
    target_dates: pd.DatetimeIndex,
    context: pd.DataFrame,
) -> tuple[list[float], list[dict[str, object]]]:
    """Roll values forward with prior-year weekday-aligned day-over-day growth."""
    ordered = history.dropna(subset=["timestamp", "target"]).copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"]).dt.normalize()
    ordered = ordered.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    values = {
        timestamp: float(target)
        for timestamp, target in zip(ordered["timestamp"], ordered["target"], strict=False)
    }
    if not values:
        return [0.0] * len(target_dates), [_empty_provenance("前一天值 × 去年日环比") for _ in target_dates]

    context_indexed = _prepare_context(context)
    fallback = float(ordered["target"].iloc[-1])
    forecasts: list[float] = []
    provenance: list[dict[str, object]] = []
    for raw_timestamp in target_dates:
        timestamp = pd.Timestamp(raw_timestamp).normalize()
        base_timestamp = timestamp - pd.Timedelta(days=1)
        base = values.get(base_timestamp)
        if base is None or not math.isfinite(base):
            base = forecasts[-1] if forecasts else fallback
        aligned, ratio_name = _align_daily_ratio_date(timestamp, context_indexed)
        comparison_timestamp = aligned - pd.Timedelta(days=1)
        raw_rate = _daily_change_rate(values, aligned)
        rate = _select_yoy_daily_change_rate(
            timestamp=timestamp,
            aligned=aligned,
            history_values=values,
            context=context_indexed,
        )
        forecast = float(base * (1.0 + rate))
        values[timestamp] = forecast
        forecasts.append(forecast)
        provenance.append(
            _ratio_provenance(
                basis="前一天值 × 去年日环比",
                base_timestamp=base_timestamp,
                base_value=base,
                prior_year_timestamp=aligned,
                prior_year_value=values.get(aligned),
                comparison_timestamp=comparison_timestamp,
                comparison_value=values.get(comparison_timestamp),
                applied_ratio=rate,
                raw_ratio=raw_rate,
                ratio_name=ratio_name,
            )
        )
    return forecasts, provenance


def _yoy_weekday_dod_anchor_forecasts(
    *,
    history: pd.DataFrame,
    target_dates: pd.DatetimeIndex,
    context: pd.DataFrame,
) -> tuple[list[float], list[dict[str, object]]]:
    """Holiday-aware DoD with independent annual-level anchors to limit drift."""
    ordered = history.dropna(subset=["timestamp", "target"]).copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"]).dt.normalize()
    ordered = ordered.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    values = {
        timestamp: float(target)
        for timestamp, target in zip(ordered["timestamp"], ordered["target"], strict=False)
    }
    if not values:
        return [0.0] * len(target_dates), [
            _empty_provenance("日环比递推与年度水平锚点融合") for _ in target_dates
        ]

    context_indexed = _prepare_context(context)
    fallback = float(ordered["target"].iloc[-1])
    level_scale = _recent_yoy_level_scale(values)
    forecasts: list[float] = []
    provenance: list[dict[str, object]] = []
    for horizon_index, raw_timestamp in enumerate(target_dates):
        timestamp = pd.Timestamp(raw_timestamp).normalize()
        base_timestamp = timestamp - pd.Timedelta(days=1)
        base = values.get(base_timestamp)
        if base is None or not math.isfinite(base):
            base = forecasts[-1] if forecasts else fallback

        aligned, ratio_name = _align_daily_ratio_date(timestamp, context_indexed)
        comparison_timestamp = aligned - pd.Timedelta(days=1)
        raw_rate = _daily_change_rate(values, aligned)
        rate = _select_yoy_daily_change_rate(
            timestamp=timestamp,
            aligned=aligned,
            history_values=values,
            context=context_indexed,
        )
        recursive_forecast = float(base * (1.0 + rate))
        anchor = _annual_level_anchor(values, aligned, level_scale)
        recursive_weight = _dod_anchor_recursive_weight(
            timestamp=timestamp,
            context=context_indexed,
            horizon_index=horizon_index,
        )
        if anchor is None or recursive_weight >= 1.0:
            forecast = recursive_forecast
            basis = "前一天值 × 去年日环比"
            used_ratio_name = ratio_name
        else:
            forecast = float(recursive_weight * recursive_forecast + (1.0 - recursive_weight) * anchor)
            basis = (
                f"日环比递推 {recursive_weight:.0%} + 年度水平锚点 "
                f"{1.0 - recursive_weight:.0%}"
            )
            used_ratio_name = f"{ratio_name} + 年度水平锚点"

        values[timestamp] = forecast
        forecasts.append(forecast)
        provenance.append(
            _ratio_provenance(
                basis=basis,
                base_timestamp=base_timestamp,
                base_value=base,
                prior_year_timestamp=aligned,
                prior_year_value=values.get(aligned),
                comparison_timestamp=comparison_timestamp,
                comparison_value=values.get(comparison_timestamp),
                applied_ratio=rate,
                raw_ratio=raw_rate,
                ratio_name=used_ratio_name,
            )
        )
    return forecasts, provenance


def _yoy_weekday_dod_anchor_v2_forecasts(
    *,
    history: pd.DataFrame,
    target_dates: pd.DatetimeIndex,
    context: pd.DataFrame,
) -> tuple[list[float], list[dict[str, object]]]:
    """Adaptive DoD/level-anchor blend learned from each series' recent history."""
    ordered = history.dropna(subset=["timestamp", "target"]).copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"]).dt.normalize()
    ordered = ordered.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    values = {
        timestamp: float(target)
        for timestamp, target in zip(ordered["timestamp"], ordered["target"], strict=False)
    }
    if not values:
        return [0.0] * len(target_dates), [
            _empty_provenance("自适应日环比递推与年度水平锚点融合") for _ in target_dates
        ]

    context_indexed = _prepare_context(context)
    fallback = float(ordered["target"].iloc[-1])
    level_scale = _recent_yoy_level_scale(values)
    recursive_weights = _learn_anchor_v2_recursive_weights(
        history_values=values,
        context=context_indexed,
        level_scale=level_scale,
    )
    forecasts: list[float] = []
    provenance: list[dict[str, object]] = []
    for raw_timestamp in target_dates:
        timestamp = pd.Timestamp(raw_timestamp).normalize()
        base_timestamp = timestamp - pd.Timedelta(days=1)
        base = values.get(base_timestamp)
        if base is None or not math.isfinite(base):
            base = forecasts[-1] if forecasts else fallback

        aligned, ratio_name = _align_daily_ratio_date(timestamp, context_indexed)
        comparison_timestamp = aligned - pd.Timedelta(days=1)
        raw_rate = _daily_change_rate(values, aligned)
        rate = _select_yoy_daily_change_rate(
            timestamp=timestamp,
            aligned=aligned,
            history_values=values,
            context=context_indexed,
        )
        recursive_forecast = float(base * (1.0 + rate))
        anchor = _annual_level_anchor(values, aligned, level_scale)
        state = _anchor_v2_state(timestamp, context_indexed)
        recursive_weight = recursive_weights[state]
        if anchor is None:
            forecast = recursive_forecast
            basis = "前一天值 × 去年日环比"
            used_ratio_name = ratio_name
        else:
            forecast = float(recursive_weight * recursive_forecast + (1.0 - recursive_weight) * anchor)
            basis = (
                f"自适应日环比递推 {recursive_weight:.0%} + 年度水平锚点 "
                f"{1.0 - recursive_weight:.0%}"
            )
            used_ratio_name = f"{ratio_name} + 自适应年度水平锚点"

        values[timestamp] = forecast
        forecasts.append(forecast)
        provenance.append(
            _ratio_provenance(
                basis=basis,
                base_timestamp=base_timestamp,
                base_value=base,
                prior_year_timestamp=aligned,
                prior_year_value=values.get(aligned),
                comparison_timestamp=comparison_timestamp,
                comparison_value=values.get(comparison_timestamp),
                applied_ratio=rate,
                raw_ratio=raw_rate,
                ratio_name=used_ratio_name,
            )
        )
    return forecasts, provenance


def _learn_anchor_v2_recursive_weights(
    *,
    history_values: dict[pd.Timestamp, float],
    context: pd.DataFrame,
    level_scale: float,
) -> dict[str, float]:
    """Learn continuous blend weights without a ratio-size trigger or replacement rule."""
    errors: dict[str, list[tuple[float, float]]] = {
        state: [] for state in ANCHOR_V2_DEFAULT_RECURSIVE_WEIGHTS
    }
    recent_dates = sorted(history_values)[-ANCHOR_V2_WEIGHT_LOOKBACK_DAYS:]
    for timestamp in recent_dates:
        actual = history_values[timestamp]
        previous = history_values.get(timestamp - pd.Timedelta(days=1))
        if actual == 0 or previous is None or not math.isfinite(actual) or not math.isfinite(previous):
            continue
        aligned, _ = _align_daily_ratio_date(timestamp, context)
        rate = _daily_change_rate(history_values, aligned)
        anchor = _annual_level_anchor(history_values, aligned, level_scale)
        if rate is None or anchor is None:
            continue
        recursive = previous * (1.0 + rate)
        if not math.isfinite(recursive):
            continue
        state = _anchor_v2_state(timestamp, context)
        errors[state].append((abs(recursive - actual) / abs(actual), abs(anchor - actual) / abs(actual)))

    weights: dict[str, float] = {}
    for state, default_weight in ANCHOR_V2_DEFAULT_RECURSIVE_WEIGHTS.items():
        samples = errors[state]
        if not samples:
            weights[state] = default_weight
            continue
        recursive_error = float(pd.Series([sample[0] for sample in samples]).median())
        anchor_error = float(pd.Series([sample[1] for sample in samples]).median())
        if recursive_error + anchor_error == 0:
            learned_weight = default_weight
        else:
            learned_weight = anchor_error / (recursive_error + anchor_error)
        sample_count = len(samples)
        weights[state] = float(
            (sample_count * learned_weight + ANCHOR_V2_WEIGHT_PRIOR_STRENGTH * default_weight)
            / (sample_count + ANCHOR_V2_WEIGHT_PRIOR_STRENGTH)
        )
    return weights


def _anchor_v2_state(timestamp: pd.Timestamp, context: pd.DataFrame) -> str:
    holiday_column = _special_holiday_column(context)
    if _holiday_code(context, timestamp, holiday_column) is not None:
        return "holiday"
    if any(
        _holiday_code(context, timestamp - pd.Timedelta(days=offset), holiday_column) is not None
        for offset in range(1, DOD_ANCHOR_POST_HOLIDAY_DAYS + 1)
    ):
        return "post_holiday"
    return "normal"


def _dod2_forecasts(
    *,
    history: pd.DataFrame,
    target_dates: pd.DatetimeIndex,
) -> tuple[list[float], list[dict[str, object]]]:
    """Weekday-aligned day-over-day baseline without holiday alignment."""

    ordered = history.dropna(subset=["timestamp", "target"]).copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"]).dt.normalize()
    ordered = ordered.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    values = {
        timestamp: float(target)
        for timestamp, target in zip(ordered["timestamp"], ordered["target"], strict=False)
    }
    if not values:
        return [0.0] * len(target_dates), [_empty_provenance("前一天值 × 去年同星期日环比") for _ in target_dates]

    fallback = float(ordered["target"].iloc[-1])
    forecasts: list[float] = []
    provenance: list[dict[str, object]] = []
    for raw_timestamp in target_dates:
        timestamp = pd.Timestamp(raw_timestamp).normalize()
        base_timestamp = timestamp - pd.Timedelta(days=1)
        base = values.get(base_timestamp)
        if base is None or not math.isfinite(base):
            base = forecasts[-1] if forecasts else fallback
        aligned = _align_to_prior_year_weekday(timestamp)
        comparison_timestamp = aligned - pd.Timedelta(days=1)
        raw_rate = _daily_change_rate(values, aligned)
        rate = raw_rate if raw_rate is not None else 0.0
        forecast = float(base * (1.0 + rate))
        values[timestamp] = forecast
        forecasts.append(forecast)
        provenance.append(
            _ratio_provenance(
                basis="前一天值 × 去年同星期日环比",
                base_timestamp=base_timestamp,
                base_value=base,
                prior_year_timestamp=aligned,
                prior_year_value=values.get(aligned),
                comparison_timestamp=comparison_timestamp,
                comparison_value=values.get(comparison_timestamp),
                applied_ratio=rate,
                raw_ratio=raw_rate,
                ratio_name="去年同星期日日环比",
            )
        )
    return forecasts, provenance


def _recent_yoy_level_scale(history_values: dict[pd.Timestamp, float]) -> float:
    """Return a robust, non-recursive current-to-prior-year level multiplier."""
    ratios: list[float] = []
    for timestamp in sorted(history_values)[-YOY_LEVEL_SCALE_LOOKBACK_DAYS:]:
        prior_year_value = history_values.get(_align_to_prior_year_weekday(timestamp))
        current_value = history_values[timestamp]
        if (
            prior_year_value is None
            or prior_year_value == 0
            or not math.isfinite(current_value)
            or not math.isfinite(prior_year_value)
        ):
            continue
        ratio = current_value / prior_year_value
        if math.isfinite(ratio) and ratio >= 0:
            ratios.append(float(ratio))
    return float(pd.Series(ratios).median()) if ratios else 1.0


def _annual_level_anchor(
    history_values: dict[pd.Timestamp, float],
    aligned_timestamp: pd.Timestamp,
    level_scale: float,
) -> float | None:
    reference_value = history_values.get(aligned_timestamp)
    if reference_value is None or not math.isfinite(reference_value):
        return None
    anchor = reference_value * level_scale
    return float(anchor) if math.isfinite(anchor) else None


def _dod_anchor_recursive_weight(
    *,
    timestamp: pd.Timestamp,
    context: pd.DataFrame,
    horizon_index: int,
) -> float:
    """Use stronger anchors for holiday days, their aftermath, and weekly resets."""
    weight = 1.0
    holiday_column = _special_holiday_column(context)
    if _holiday_code(context, timestamp, holiday_column) is not None:
        weight = DOD_ANCHOR_HOLIDAY_RECURSIVE_WEIGHT
    elif any(
        _holiday_code(context, timestamp - pd.Timedelta(days=offset), holiday_column) is not None
        for offset in range(1, DOD_ANCHOR_POST_HOLIDAY_DAYS + 1)
    ):
        weight = DOD_ANCHOR_POST_HOLIDAY_RECURSIVE_WEIGHT
    if (horizon_index + 1) % 7 == 0:
        weight = min(weight, DOD_ANCHOR_WEEKLY_RECURSIVE_WEIGHT)
    return weight


def _select_yoy_weekly_ratio(
    *,
    timestamp: pd.Timestamp,
    history_values: dict[pd.Timestamp, float],
    context: pd.DataFrame,
) -> float:
    aligned = _align_to_prior_year_weekday(timestamp)
    raw_ratio = _weekly_ratio(history_values, aligned)
    if raw_ratio is not None:
        return raw_ratio

    condition_columns = _condition_columns(context)
    target_conditions = _condition_values(context, timestamp, condition_columns)
    previous_conditions = _condition_values(
        context,
        timestamp - pd.Timedelta(days=7),
        condition_columns,
    )
    candidates = _ratio_candidates(
        aligned=aligned,
        history_values=history_values,
        context=context,
        condition_columns=condition_columns,
        target_conditions=target_conditions,
        previous_conditions=previous_conditions,
        same_month_only=True,
    )
    if len(candidates) < 2:
        candidates = _ratio_candidates(
            aligned=aligned,
            history_values=history_values,
            context=context,
            condition_columns=condition_columns,
            target_conditions=target_conditions,
            previous_conditions=previous_conditions,
            same_month_only=False,
    )
    reference_ratio = _surrounding_ratio_median(candidates, aligned)
    return reference_ratio if reference_ratio is not None else 1.0


def _select_yoy_weekly_ratio_original(
    *,
    timestamp: pd.Timestamp,
    history_values: dict[pd.Timestamp, float],
    context: pd.DataFrame,
) -> float:
    aligned = _align_to_prior_year_weekday(timestamp)
    raw_ratio = _weekly_ratio(history_values, aligned)
    if raw_ratio is not None:
        return raw_ratio

    condition_columns = _condition_columns(context)
    target_conditions = _condition_values(context, timestamp, condition_columns)
    previous_conditions = _condition_values(
        context,
        timestamp - pd.Timedelta(days=7),
        condition_columns,
    )

    candidates = _ratio_candidates(
        aligned=aligned,
        history_values=history_values,
        context=context,
        condition_columns=condition_columns,
        target_conditions=target_conditions,
        previous_conditions=previous_conditions,
        same_month_only=True,
    )
    if len(candidates) < 2:
        candidates = _ratio_candidates(
            aligned=aligned,
            history_values=history_values,
            context=context,
            condition_columns=condition_columns,
            target_conditions=target_conditions,
            previous_conditions=previous_conditions,
            same_month_only=False,
    )
    reference_ratio = _surrounding_ratio_median(candidates, aligned)
    return reference_ratio if reference_ratio is not None else 1.0


def _select_yoy_daily_change_rate(
    *,
    timestamp: pd.Timestamp,
    aligned: pd.Timestamp,
    history_values: dict[pd.Timestamp, float],
    context: pd.DataFrame,
) -> float:
    raw_rate = _daily_change_rate(history_values, aligned)
    if raw_rate is not None:
        return raw_rate

    condition_columns = _condition_columns(context)
    target_conditions = _condition_values(context, timestamp, condition_columns)
    previous_conditions = _condition_values(
        context,
        timestamp - pd.Timedelta(days=1),
        condition_columns,
    )
    candidates = _daily_change_candidates(
        aligned=aligned,
        history_values=history_values,
        context=context,
        condition_columns=condition_columns,
        target_conditions=target_conditions,
        previous_conditions=previous_conditions,
        same_month_only=True,
    )
    if len(candidates) < 2:
        candidates = _daily_change_candidates(
            aligned=aligned,
            history_values=history_values,
            context=context,
            condition_columns=condition_columns,
            target_conditions=target_conditions,
            previous_conditions=previous_conditions,
            same_month_only=False,
    )
    reference_rate = _daily_change_reference_median(candidates, aligned)
    return reference_rate if reference_rate is not None else 0.0


def _align_to_prior_year_weekday(timestamp: pd.Timestamp) -> pd.Timestamp:
    anchor = (timestamp - pd.DateOffset(years=1)).normalize()
    weekday_delta = ((timestamp.weekday() - anchor.weekday() + 3) % 7) - 3
    return anchor + pd.Timedelta(days=weekday_delta)


def _align_daily_ratio_date(
    timestamp: pd.Timestamp,
    context: pd.DataFrame,
) -> tuple[pd.Timestamp, str]:
    """Choose a daily-ratio reference date, preferring matching holiday codes."""

    weekday_aligned = _align_to_prior_year_weekday(timestamp)
    holiday_column = _special_holiday_column(context)
    holiday_code = _holiday_code(context, timestamp, holiday_column)
    if holiday_column is None or holiday_code is None:
        return weekday_aligned, "去年对齐日日环比"

    for years_back in (1, 2):
        candidate_year = timestamp.year - years_back
        matches = _holiday_dates(
            context,
            holiday_column=holiday_column,
            year=candidate_year,
            holiday_code=holiday_code,
        )
        if matches:
            label = "去年同假期日环比" if years_back == 1 else "前年同假期日环比"
            return _nearest_date(matches, timestamp - pd.DateOffset(years=years_back)), label

    day_number = _holiday_day_number(holiday_code)
    if day_number is not None:
        prior_year_holidays = [
            date
            for date in context.index
            if date.year == timestamp.year - 1
            and _holiday_day_number(_holiday_code(context, date, holiday_column)) == day_number
        ]
        if prior_year_holidays:
            return (
                _nearest_date(prior_year_holidays, timestamp - pd.DateOffset(years=1)),
                "去年最近同假期天数日环比",
            )

    return weekday_aligned, "去年对齐日日环比"


def _special_holiday_column(context: pd.DataFrame) -> str | None:
    return next(
        (
            str(column)
            for column in context.columns
            if str(column).strip() == "特殊假期"
        ),
        None,
    )


def _holiday_code(
    context: pd.DataFrame,
    timestamp: pd.Timestamp,
    holiday_column: str | None,
) -> str | None:
    if holiday_column is None or timestamp not in context.index:
        return None
    value = context.loc[timestamp].get(holiday_column)
    if pd.isna(value):
        return None
    code = str(value).strip()
    return None if not code or code.casefold() == "none" else code


def _holiday_dates(
    context: pd.DataFrame,
    *,
    holiday_column: str,
    year: int,
    holiday_code: str,
) -> list[pd.Timestamp]:
    return [
        date
        for date in context.index
        if date.year == year and _holiday_code(context, date, holiday_column) == holiday_code
    ]


def _holiday_day_number(holiday_code: str | None) -> int | None:
    if holiday_code is None:
        return None
    match = re.search(r"(\d+)$", holiday_code)
    return int(match.group(1)) if match else None


def _nearest_date(dates: list[pd.Timestamp], target: pd.Timestamp) -> pd.Timestamp:
    return min(dates, key=lambda date: (abs((date - target).days), date))


def _weekly_ratio(
    history_values: dict[pd.Timestamp, float],
    timestamp: pd.Timestamp,
) -> float | None:
    current = history_values.get(timestamp)
    previous = history_values.get(timestamp - pd.Timedelta(days=7))
    if current is None or previous is None or previous == 0:
        return None
    ratio = current / previous
    return float(ratio) if math.isfinite(ratio) and ratio >= 0 else None


def _daily_change_rate(
    history_values: dict[pd.Timestamp, float],
    timestamp: pd.Timestamp,
) -> float | None:
    current = history_values.get(timestamp)
    previous = history_values.get(timestamp - pd.Timedelta(days=1))
    if current is None or previous is None or previous == 0:
        return None
    factor = current / previous
    if not math.isfinite(factor) or factor < 0:
        return None
    return float(factor - 1.0)


def _daily_change_candidates(
    *,
    aligned: pd.Timestamp,
    history_values: dict[pd.Timestamp, float],
    context: pd.DataFrame,
    condition_columns: list[str],
    target_conditions: dict[str, object],
    previous_conditions: dict[str, object],
    same_month_only: bool,
) -> list[tuple[pd.Timestamp, float]]:
    candidates: list[tuple[pd.Timestamp, float]] = []
    for week_offset in range(-YOY_RATIO_SEARCH_WEEKS, YOY_RATIO_SEARCH_WEEKS + 1):
        if week_offset == 0:
            continue
        candidate = aligned + pd.Timedelta(days=7 * week_offset)
        if same_month_only and candidate.month != aligned.month:
            continue
        rate = _daily_change_rate(history_values, candidate)
        if rate is None:
            continue
        if not _conditions_match(
            context,
            candidate,
            condition_columns,
            target_conditions,
        ):
            continue
        if not _conditions_match(
            context,
            candidate - pd.Timedelta(days=1),
            condition_columns,
            previous_conditions,
        ):
            continue
        candidates.append((candidate, rate))
    return candidates


def _daily_change_reference_median(
    candidates: list[tuple[pd.Timestamp, float]],
    aligned: pd.Timestamp,
) -> float | None:
    return _surrounding_value_median(candidates, aligned)


def _ratio_candidates(
    *,
    aligned: pd.Timestamp,
    history_values: dict[pd.Timestamp, float],
    context: pd.DataFrame,
    condition_columns: list[str],
    target_conditions: dict[str, object],
    previous_conditions: dict[str, object],
    same_month_only: bool,
) -> list[tuple[pd.Timestamp, float]]:
    candidates: list[tuple[pd.Timestamp, float]] = []
    for week_offset in range(-YOY_RATIO_SEARCH_WEEKS, YOY_RATIO_SEARCH_WEEKS + 1):
        if week_offset == 0:
            continue
        candidate = aligned + pd.Timedelta(days=7 * week_offset)
        if same_month_only and candidate.month != aligned.month:
            continue
        ratio = _weekly_ratio(history_values, candidate)
        if ratio is None:
            continue
        if not _conditions_match(
            context,
            candidate,
            condition_columns,
            target_conditions,
        ):
            continue
        if not _conditions_match(
            context,
            candidate - pd.Timedelta(days=7),
            condition_columns,
            previous_conditions,
        ):
            continue
        candidates.append((candidate, ratio))
    return candidates


def _surrounding_ratio_median(
    candidates: list[tuple[pd.Timestamp, float]],
    aligned: pd.Timestamp,
) -> float | None:
    return _surrounding_value_median(candidates, aligned)


def _surrounding_value_median(
    candidates: list[tuple[pd.Timestamp, float]],
    aligned: pd.Timestamp,
) -> float | None:
    before = sorted(
        (candidate for candidate in candidates if candidate[0] < aligned),
        key=lambda candidate: candidate[0],
        reverse=True,
    )[:2]
    after = sorted(
        (candidate for candidate in candidates if candidate[0] > aligned),
        key=lambda candidate: candidate[0],
    )[:2]
    surrounding = before + after
    if not surrounding:
        surrounding = sorted(
            candidates,
            key=lambda candidate: abs((candidate[0] - aligned).days),
        )[:4]
    if not surrounding:
        return None
    return float(pd.Series([ratio for _, ratio in surrounding]).median())


def _relative_factor_deviation(raw_factor: float, reference_factor: float) -> float:
    if reference_factor == 0:
        return 0.0 if raw_factor == 0 else math.inf
    deviation = abs(raw_factor / reference_factor - 1.0)
    return deviation if math.isfinite(deviation) else math.inf


def _prepare_context(context: pd.DataFrame) -> pd.DataFrame:
    prepared = context.dropna(subset=["timestamp"]).copy()
    prepared["timestamp"] = pd.to_datetime(prepared["timestamp"]).dt.normalize()
    return (
        prepared.sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .set_index("timestamp")
    )


def _condition_columns(context: pd.DataFrame) -> list[str]:
    tokens = (*WEATHER_COLUMN_TOKENS, *HOLIDAY_COLUMN_TOKENS)
    return [
        column
        for column in context.columns
        if column not in {"item_id", "timestamp", "target"}
        and any(token in str(column).lower() for token in tokens)
    ]


def _condition_values(
    context: pd.DataFrame,
    timestamp: pd.Timestamp,
    columns: list[str],
) -> dict[str, object]:
    if timestamp not in context.index:
        return {}
    row = context.loc[timestamp]
    return {
        column: row[column] for column in columns if column in row.index and pd.notna(row[column])
    }


def _conditions_match(
    context: pd.DataFrame,
    timestamp: pd.Timestamp,
    columns: list[str],
    expected: dict[str, object],
) -> bool:
    if not expected or timestamp not in context.index:
        return not expected
    row = context.loc[timestamp]
    for column in columns:
        if column not in expected:
            continue
        actual = row.get(column)
        if pd.isna(actual) or actual != expected[column]:
            return False
    return True


def _mtd_forecasts(series: pd.DataFrame, future_dates: pd.DatetimeIndex) -> list[float]:
    history = series.copy()
    history["timestamp"] = pd.to_datetime(history["timestamp"])
    forecasts = []
    for ts in future_dates:
        month_start = ts.replace(day=1)
        mtd_history = history[(history["timestamp"] >= month_start) & (history["timestamp"] < ts)]
        if not mtd_history.empty:
            forecasts.append(float(mtd_history["target"].mean()))
        else:
            same_month = history[history["timestamp"].dt.month == ts.month]
            forecasts.append(
                float(same_month["target"].mean())
                if not same_month.empty
                else float(history["target"].iloc[-1])
            )
    return forecasts


def _prediction_frame(
    model: str,
    item_id: str,
    actual: pd.DataFrame,
    forecasts: list[float],
    window_index: int,
    provenance: list[dict[str, object]] | None = None,
) -> pd.DataFrame:
    actual_values = pd.to_numeric(actual["target"], errors="coerce")
    if provenance is None:
        provenance = [{} for _ in forecasts]
    if len(provenance) != len(forecasts):
        raise ValueError("provenance length must match forecasts")
    frame = pd.DataFrame(
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
    for column in PREDICTION_PROVENANCE_COLUMNS:
        frame[column] = [row.get(column, pd.NA) for row in provenance]
    return frame


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
        *PREDICTION_PROVENANCE_COLUMNS,
    ]


def _empty_provenance(basis: str) -> dict[str, object]:
    return {"预测依据": basis, "环比来源": "历史不足，使用默认值"}


def _ratio_provenance(
    *,
    basis: str,
    base_timestamp: pd.Timestamp,
    base_value: float,
    prior_year_timestamp: pd.Timestamp,
    prior_year_value: float | None,
    comparison_timestamp: pd.Timestamp,
    comparison_value: float | None,
    applied_ratio: float,
    raw_ratio: float | None,
    ratio_name: str,
) -> dict[str, object]:
    used_raw_ratio = raw_ratio is not None and math.isclose(applied_ratio, raw_ratio)
    return {
        "预测依据": basis,
        "预测基准日期": base_timestamp,
        "预测基准值": base_value,
        "去年环比日期": prior_year_timestamp,
        "去年环比值": prior_year_value,
        "去年环比对比日期": comparison_timestamp,
        "去年环比对比值": comparison_value,
        "实际采用环比": applied_ratio,
        "环比来源": ratio_name if used_raw_ratio else f"{ratio_name}稳健参考",
    }
