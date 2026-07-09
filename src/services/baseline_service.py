from __future__ import annotations

import math

import pandas as pd

from src.core.constants import BASELINE_MODEL_NAMES, ROLLING_MEAN_WINDOW, SEASONAL_LAG

YOY_WEEKDAY_WOW_MODEL = "YoY Weekday WoW"
YOY_WEEKDAY_WOW_ORIGINAL_MODEL = "YoY Weekday WoW Original"
YOY_WEEKDAY_WOW_HYBRID_MODEL = "YoY Weekday WoW Hybrid"
YOY_WEEKDAY_DOD_MODEL = "YoY Weekday DoD"
BASELINE_MODELS = BASELINE_MODEL_NAMES

YOY_LAG = {"M": 12, "W": 52, "D": 365}
WOW_LAG = {"M": 1, "W": 1, "D": 7}
YOY_RATIO_SEARCH_WEEKS = 4
YOY_WEEKDAY_ROBUST_DEVIATION_THRESHOLD = 0.25
YOY_WEEKDAY_WOW_ORIGINAL_VOLATILITY_THRESHOLD = 0.25
YOY_WEEKDAY_HYBRID_ORIGINAL_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
YOY_WEEKDAY_HYBRID_VALIDATION_WINDOWS = 3
WEATHER_COLUMN_TOKENS = ("天气", "雨雪", "weather", "rain", "snow")
HOLIDAY_COLUMN_TOKENS = ("节假日", "假期", "holiday")


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
    selected_models = tuple(models or _baseline_models_for_freq(freq))

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
            YOY_WEEKDAY_WOW_ORIGINAL_MODEL,
            YOY_WEEKDAY_WOW_HYBRID_MODEL,
            YOY_WEEKDAY_DOD_MODEL,
        )
    return common


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
        return _predict_yoy_weekday_wow(item_id, actual, train, window_index)
    if model == YOY_WEEKDAY_WOW_ORIGINAL_MODEL and freq == "D":
        return _predict_yoy_weekday_wow_original(item_id, actual, train, window_index)
    if model == YOY_WEEKDAY_WOW_HYBRID_MODEL and freq == "D":
        return _predict_yoy_weekday_wow_hybrid(item_id, actual, train, window_index)
    if model == YOY_WEEKDAY_DOD_MODEL and freq == "D":
        return _predict_yoy_weekday_dod(item_id, actual, train, window_index)
    return None


def generate_baseline_future_forecast(
    *,
    data: pd.DataFrame,
    freq: str,
    prediction_length: int,
    model: str,
) -> pd.DataFrame:
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
            forecasts = _yoy_weekday_wow_forecasts(
                history=series,
                target_dates=future_dates,
                context=context,
            )
        elif model == YOY_WEEKDAY_WOW_ORIGINAL_MODEL:
            forecasts = _yoy_weekday_wow_original_forecasts(
                history=series,
                target_dates=future_dates,
                context=context,
            )
        elif model == YOY_WEEKDAY_WOW_HYBRID_MODEL:
            forecasts = _yoy_weekday_wow_hybrid_forecasts(
                history=series,
                target_dates=future_dates,
                context=context,
            )
        elif model == YOY_WEEKDAY_DOD_MODEL:
            forecasts = _yoy_weekday_dod_forecasts(
                history=series,
                target_dates=future_dates,
                context=context,
            )
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


def _predict_yoy_weekday_wow(
    item_id: str,
    actual: pd.DataFrame,
    train: pd.DataFrame,
    window_index: int,
) -> pd.DataFrame:
    forecasts = _yoy_weekday_wow_forecasts(
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
    )


def _predict_yoy_weekday_wow_original(
    item_id: str,
    actual: pd.DataFrame,
    train: pd.DataFrame,
    window_index: int,
) -> pd.DataFrame:
    forecasts = _yoy_weekday_wow_original_forecasts(
        history=train,
        target_dates=pd.DatetimeIndex(pd.to_datetime(actual["timestamp"])),
        context=pd.concat([train, actual], ignore_index=True),
    )
    return _prediction_frame(
        YOY_WEEKDAY_WOW_ORIGINAL_MODEL,
        item_id,
        actual,
        forecasts,
        window_index,
    )


def _predict_yoy_weekday_wow_hybrid(
    item_id: str,
    actual: pd.DataFrame,
    train: pd.DataFrame,
    window_index: int,
) -> pd.DataFrame:
    forecasts = _yoy_weekday_wow_hybrid_forecasts(
        history=train,
        target_dates=pd.DatetimeIndex(pd.to_datetime(actual["timestamp"])),
        context=pd.concat([train, actual], ignore_index=True),
    )
    return _prediction_frame(
        YOY_WEEKDAY_WOW_HYBRID_MODEL,
        item_id,
        actual,
        forecasts,
        window_index,
    )


def _predict_yoy_weekday_dod(
    item_id: str,
    actual: pd.DataFrame,
    train: pd.DataFrame,
    window_index: int,
) -> pd.DataFrame:
    forecasts = _yoy_weekday_dod_forecasts(
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
    )


def _lagged_forecasts(
    history: pd.DataFrame,
    horizon: int,
    lag: int,
    fallback: float,
) -> list[float]:
    values = list(history["target"].astype(float))
    forecasts: list[float] = []
    for horizon_index in range(horizon):
        forecasts.append(_seasonal_value(values, forecasts, lag, horizon_index, fallback))
    return forecasts


def _yoy_weekday_wow_forecasts(
    *,
    history: pd.DataFrame,
    target_dates: pd.DatetimeIndex,
    context: pd.DataFrame,
) -> list[float]:
    """Roll current-year values forward with prior-year weekday-aligned WoW ratios.

    A target date is first anchored to the same natural calendar date one year
    earlier, then shifted to the nearest matching weekday.  The prior-year
    week-over-week ratio is applied to the target date's current-year value from
    seven days earlier.  Forecasts therefore become the base for the next week.
    """
    ordered = history.dropna(subset=["timestamp", "target"]).copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"]).dt.normalize()
    ordered = ordered.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    values = {
        timestamp: float(target)
        for timestamp, target in zip(ordered["timestamp"], ordered["target"], strict=False)
    }
    if not values:
        return [0.0] * len(target_dates)

    context_indexed = _prepare_context(context)
    fallback = float(ordered["target"].iloc[-1])
    forecasts: list[float] = []
    for raw_timestamp in target_dates:
        timestamp = pd.Timestamp(raw_timestamp).normalize()
        base = values.get(timestamp - pd.Timedelta(days=7))
        if base is None or not math.isfinite(base):
            base = forecasts[-7] if len(forecasts) >= 7 else fallback
        ratio = _select_yoy_weekly_ratio(
            timestamp=timestamp,
            history_values=values,
            context=context_indexed,
        )
        forecast = float(base * ratio)
        values[timestamp] = forecast
        forecasts.append(forecast)
    return forecasts


def _yoy_weekday_wow_original_forecasts(
    *,
    history: pd.DataFrame,
    target_dates: pd.DatetimeIndex,
    context: pd.DataFrame,
) -> list[float]:
    """Original weekday-aligned YoY WoW baseline from record.py."""
    ordered = history.dropna(subset=["timestamp", "target"]).copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"]).dt.normalize()
    ordered = ordered.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    values = {
        timestamp: float(target)
        for timestamp, target in zip(ordered["timestamp"], ordered["target"], strict=False)
    }
    if not values:
        return [0.0] * len(target_dates)

    context_indexed = _prepare_context(context)
    fallback = float(ordered["target"].iloc[-1])
    forecasts: list[float] = []
    for raw_timestamp in target_dates:
        timestamp = pd.Timestamp(raw_timestamp).normalize()
        base = values.get(timestamp - pd.Timedelta(days=7))
        if base is None or not math.isfinite(base):
            base = forecasts[-7] if len(forecasts) >= 7 else fallback
        ratio = _select_yoy_weekly_ratio_original(
            timestamp=timestamp,
            history_values=values,
            context=context_indexed,
        )
        forecast = float(base * ratio)
        values[timestamp] = forecast
        forecasts.append(forecast)
    return forecasts


def _yoy_weekday_wow_hybrid_forecasts(
    *,
    history: pd.DataFrame,
    target_dates: pd.DatetimeIndex,
    context: pd.DataFrame,
) -> list[float]:
    ordered = history.dropna(subset=["timestamp", "target"]).copy()
    if ordered.empty:
        return [0.0] * len(target_dates)
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"]).dt.normalize()
    ordered = ordered.sort_values("timestamp").drop_duplicates("timestamp", keep="last")

    original_weight = _select_yoy_weekday_wow_hybrid_weight(
        history=ordered,
        context=context,
        prediction_length=len(target_dates),
    )
    original_forecasts = _yoy_weekday_wow_original_forecasts(
        history=ordered,
        target_dates=target_dates,
        context=context,
    )
    fallback = float(ordered["target"].iloc[-1])
    seasonal_forecasts = _lagged_forecasts(
        ordered,
        len(target_dates),
        SEASONAL_LAG["D"],
        fallback,
    )
    return [
        float(original_weight * original + (1.0 - original_weight) * seasonal)
        for original, seasonal in zip(original_forecasts, seasonal_forecasts, strict=False)
    ]


def _select_yoy_weekday_wow_hybrid_weight(
    *,
    history: pd.DataFrame,
    context: pd.DataFrame,
    prediction_length: int,
) -> float:
    validation_length = min(max(1, prediction_length), 30)
    scores = {weight: 0.0 for weight in YOY_WEEKDAY_HYBRID_ORIGINAL_WEIGHTS}
    evaluated_windows = 0

    for window_index in range(YOY_WEEKDAY_HYBRID_VALIDATION_WINDOWS):
        validation_end = len(history) - window_index * validation_length
        validation_start = validation_end - validation_length
        if validation_start <= 0:
            continue
        train = history.iloc[:validation_start].copy()
        actual = history.iloc[validation_start:validation_end].copy()
        if train.empty or actual.empty:
            continue

        target_dates = pd.DatetimeIndex(pd.to_datetime(actual["timestamp"]))
        validation_context = _validation_context_for_dates(
            context=context,
            train=train,
            actual=actual,
        )
        original_forecasts = _yoy_weekday_wow_original_forecasts(
            history=train,
            target_dates=target_dates,
            context=validation_context,
        )
        fallback = float(train["target"].iloc[-1])
        seasonal_forecasts = _lagged_forecasts(
            train,
            len(target_dates),
            SEASONAL_LAG["D"],
            fallback,
        )
        actual_values = pd.to_numeric(actual["target"], errors="coerce").astype(float).to_list()
        for weight in YOY_WEEKDAY_HYBRID_ORIGINAL_WEIGHTS:
            hybrid_forecasts = [
                float(weight * original + (1.0 - weight) * seasonal)
                for original, seasonal in zip(
                    original_forecasts,
                    seasonal_forecasts,
                    strict=False,
                )
            ]
            scores[weight] += _wape(actual_values, hybrid_forecasts)
        evaluated_windows += 1

    if evaluated_windows == 0:
        return 0.5
    return min(
        YOY_WEEKDAY_HYBRID_ORIGINAL_WEIGHTS,
        key=lambda weight: (scores[weight] / evaluated_windows, abs(weight - 0.5)),
    )


def _validation_context_for_dates(
    *,
    context: pd.DataFrame,
    train: pd.DataFrame,
    actual: pd.DataFrame,
) -> pd.DataFrame:
    if context.empty:
        return pd.concat([train, actual], ignore_index=True)
    timestamps = pd.to_datetime(context["timestamp"]).dt.normalize()
    max_timestamp = pd.Timestamp(actual["timestamp"].max()).normalize()
    return context.loc[timestamps.le(max_timestamp)].copy()


def _wape(actual_values: list[float], forecast_values: list[float]) -> float:
    pairs = [
        (actual, forecast)
        for actual, forecast in zip(actual_values, forecast_values, strict=False)
        if math.isfinite(actual) and math.isfinite(forecast)
    ]
    if not pairs:
        return math.inf
    denominator = sum(abs(actual) for actual, _ in pairs)
    numerator = sum(abs(forecast - actual) for actual, forecast in pairs)
    return numerator / denominator if denominator else numerator / len(pairs)


def _yoy_weekday_dod_forecasts(
    *,
    history: pd.DataFrame,
    target_dates: pd.DatetimeIndex,
    context: pd.DataFrame,
) -> list[float]:
    """Roll values forward with prior-year weekday-aligned day-over-day growth."""
    ordered = history.dropna(subset=["timestamp", "target"]).copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"]).dt.normalize()
    ordered = ordered.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    values = {
        timestamp: float(target)
        for timestamp, target in zip(ordered["timestamp"], ordered["target"], strict=False)
    }
    if not values:
        return [0.0] * len(target_dates)

    context_indexed = _prepare_context(context)
    fallback = float(ordered["target"].iloc[-1])
    forecasts: list[float] = []
    for raw_timestamp in target_dates:
        timestamp = pd.Timestamp(raw_timestamp).normalize()
        base = values.get(timestamp - pd.Timedelta(days=1))
        if base is None or not math.isfinite(base):
            base = forecasts[-1] if forecasts else fallback
        rate = _select_yoy_daily_change_rate(
            timestamp=timestamp,
            history_values=values,
            context=context_indexed,
        )
        forecast = float(base * (1.0 + rate))
        values[timestamp] = forecast
        forecasts.append(forecast)
    return forecasts


def _select_yoy_weekly_ratio(
    *,
    timestamp: pd.Timestamp,
    history_values: dict[pd.Timestamp, float],
    context: pd.DataFrame,
) -> float:
    aligned = _align_to_prior_year_weekday(timestamp)
    raw_ratio = _weekly_ratio(history_values, aligned)
    condition_columns = _condition_columns(context)
    target_conditions = _condition_values(context, timestamp, condition_columns)
    previous_conditions = _condition_values(
        context,
        timestamp - pd.Timedelta(days=7),
        condition_columns,
    )
    raw_condition_mismatch = not _conditions_match(
        context,
        aligned,
        condition_columns,
        target_conditions,
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
    if raw_ratio is None:
        return reference_ratio if reference_ratio is not None else 1.0
    if reference_ratio is None or reference_ratio == 0:
        return raw_ratio
    ratio_deviation = _relative_factor_deviation(raw_ratio, reference_ratio)
    if (
        raw_condition_mismatch
        or ratio_deviation > YOY_WEEKDAY_ROBUST_DEVIATION_THRESHOLD
    ):
        return reference_ratio
    return raw_ratio


def _select_yoy_weekly_ratio_original(
    *,
    timestamp: pd.Timestamp,
    history_values: dict[pd.Timestamp, float],
    context: pd.DataFrame,
) -> float:
    aligned = _align_to_prior_year_weekday(timestamp)
    raw_ratio = _weekly_ratio(history_values, aligned)
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
    if raw_ratio is None:
        return reference_ratio if reference_ratio is not None else 1.0
    if reference_ratio is None or reference_ratio == 0:
        return raw_ratio
    relative_deviation = abs(raw_ratio / reference_ratio - 1.0)
    if relative_deviation > YOY_WEEKDAY_WOW_ORIGINAL_VOLATILITY_THRESHOLD:
        return reference_ratio
    return raw_ratio


def _select_yoy_daily_change_rate(
    *,
    timestamp: pd.Timestamp,
    history_values: dict[pd.Timestamp, float],
    context: pd.DataFrame,
) -> float:
    aligned = _align_to_prior_year_weekday(timestamp)
    raw_rate = _daily_change_rate(history_values, aligned)
    condition_columns = _condition_columns(context)
    target_conditions = _condition_values(context, timestamp, condition_columns)
    previous_conditions = _condition_values(
        context,
        timestamp - pd.Timedelta(days=1),
        condition_columns,
    )
    raw_condition_mismatch = not _conditions_match(
        context,
        aligned,
        condition_columns,
        target_conditions,
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
    if raw_rate is None:
        return reference_rate if reference_rate is not None else 0.0
    if reference_rate is None:
        return raw_rate
    rate_deviation = _relative_factor_deviation(1.0 + raw_rate, 1.0 + reference_rate)
    if (
        raw_condition_mismatch
        or rate_deviation > YOY_WEEKDAY_ROBUST_DEVIATION_THRESHOLD
    ):
        return reference_rate
    return raw_rate


def _align_to_prior_year_weekday(timestamp: pd.Timestamp) -> pd.Timestamp:
    anchor = (timestamp - pd.DateOffset(years=1)).normalize()
    weekday_delta = ((timestamp.weekday() - anchor.weekday() + 3) % 7) - 3
    return anchor + pd.Timedelta(days=weekday_delta)


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
