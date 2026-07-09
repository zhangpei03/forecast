import pandas as pd
import pytest

from src.services.baseline_service import (
    YOY_WEEKDAY_DOD_MODEL,
    YOY_WEEKDAY_WOW_HYBRID_MODEL,
    YOY_WEEKDAY_WOW_ORIGINAL_MODEL,
    YOY_WEEKDAY_WOW_MODEL,
    _align_to_prior_year_weekday,
    _select_yoy_weekday_wow_hybrid_weight,
    generate_baseline_backtest_predictions,
    generate_baseline_future_forecast,
)


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


def test_daily_baseline_includes_yoy_wow_mtd() -> None:
    dates = pd.date_range("2024-01-01", periods=400, freq="D")
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

    models = set(predictions["model"])
    assert "YoY" in models
    assert "WoW" in models
    assert "MTD Daily Avg" in models
    assert YOY_WEEKDAY_WOW_ORIGINAL_MODEL in models
    assert YOY_WEEKDAY_WOW_HYBRID_MODEL in models
    assert YOY_WEEKDAY_DOD_MODEL in models

    # WoW: lag=7, so first forecast = train[-7]
    wow = predictions[predictions["model"] == "WoW"].sort_values("timestamp")
    assert len(wow) == 30
    # train ends at index 369 (0-based), wow horizon 0 looks at 370-7=363 => value 364
    assert list(wow["forecast_p50"].head(1)) == [364.0]

    # YoY: lag=365, train has 370 rows so horizon 0 looks at 370-365=5 => value 6
    yoy = predictions[predictions["model"] == "YoY"].sort_values("timestamp")
    assert len(yoy) == 30
    assert list(yoy["forecast_p50"].head(1)) == [6.0]


def test_monthly_baseline_does_not_include_yoy_wow_mtd() -> None:
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
        num_windows=1,
    )

    models = set(predictions["model"])
    assert "YoY" not in models
    assert "WoW" not in models
    assert "MTD Daily Avg" not in models


def test_baseline_backtest_can_run_single_selected_model() -> None:
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
        num_windows=1,
        models=["Rolling Mean"],
    )

    assert set(predictions["model"]) == {"Rolling Mean"}


def test_prior_year_alignment_uses_nearest_matching_weekday() -> None:
    aligned = _align_to_prior_year_weekday(pd.Timestamp("2026-07-02"))

    assert aligned == pd.Timestamp("2025-07-03")
    assert aligned.weekday() == pd.Timestamp("2026-07-02").weekday()


def test_yoy_weekday_wow_applies_prior_year_ratio_to_current_week() -> None:
    dates = pd.date_range("2025-06-01", "2026-07-02", freq="D")
    targets = pd.Series(100.0, index=dates)
    targets.loc[pd.Timestamp("2025-06-26")] = 100.0
    targets.loc[pd.Timestamp("2025-07-03")] = 120.0
    targets.loc[pd.Timestamp("2025-07-10")] = 144.0
    targets.loc[pd.Timestamp("2025-07-17")] = 172.8
    targets.loc[pd.Timestamp("2026-06-25")] = 200.0
    targets.loc[pd.Timestamp("2026-07-02")] = 999.0
    data = pd.DataFrame(
        {
            "item_id": "A",
            "timestamp": dates,
            "target": targets.to_numpy(),
        }
    )

    predictions = generate_baseline_backtest_predictions(
        data=data,
        freq="D",
        prediction_length=1,
        num_windows=1,
    )
    selected = predictions[predictions["model"].eq(YOY_WEEKDAY_WOW_MODEL)]

    assert selected["forecast_p50"].tolist() == pytest.approx([240.0])


def test_yoy_weekday_wow_rolls_forecast_into_following_week() -> None:
    history_dates = pd.date_range("2025-06-01", "2026-07-01", freq="D")
    targets = pd.Series(100.0, index=history_dates)
    targets.loc[pd.Timestamp("2025-06-26")] = 100.0
    targets.loc[pd.Timestamp("2025-07-03")] = 110.0
    targets.loc[pd.Timestamp("2025-07-10")] = 121.0
    targets.loc[pd.Timestamp("2026-06-25")] = 200.0
    history = pd.DataFrame(
        {
            "item_id": "A",
            "timestamp": history_dates,
            "target": targets.to_numpy(),
        }
    )

    future = generate_baseline_future_forecast(
        data=history,
        freq="D",
        prediction_length=8,
        model=YOY_WEEKDAY_WOW_MODEL,
    ).sort_values("timestamp")

    assert future.iloc[0]["forecast_p50"] == pytest.approx(220.0)
    assert future.iloc[7]["forecast_p50"] == pytest.approx(242.0)


def test_yoy_weekday_wow_replaces_volatile_ratio_even_when_conditions_match() -> None:
    history_dates = pd.date_range("2025-06-01", "2026-07-15", freq="D")
    targets = pd.Series(100.0, index=history_dates)
    targets.loc[pd.Timestamp("2025-07-17")] = 200.0
    targets.loc[pd.Timestamp("2026-07-09")] = 200.0
    history = pd.DataFrame(
        {
            "item_id": "A",
            "timestamp": history_dates,
            "target": targets.to_numpy(),
        }
    )

    future = generate_baseline_future_forecast(
        data=history,
        freq="D",
        prediction_length=1,
        model=YOY_WEEKDAY_WOW_MODEL,
    )

    assert future["forecast_p50"].tolist() == pytest.approx([200.0])


def test_yoy_weekday_wow_original_uses_record_py_volatility_rule() -> None:
    history_dates = pd.date_range("2025-06-01", "2026-07-01", freq="D")
    targets = pd.Series(100.0, index=history_dates)
    weather = pd.Series(0, index=history_dates, dtype=int)

    targets.loc[pd.Timestamp("2025-06-26")] = 100.0
    targets.loc[pd.Timestamp("2025-07-03")] = 200.0
    for date in (pd.Timestamp("2025-06-19"), pd.Timestamp("2025-07-17")):
        targets.loc[date - pd.Timedelta(days=7)] = 100.0
        targets.loc[date] = 110.0
        weather.loc[date - pd.Timedelta(days=7)] = 1
        weather.loc[date] = 1
    targets.loc[pd.Timestamp("2026-06-25")] = 300.0
    weather.loc[pd.Timestamp("2026-06-25")] = 1

    history = pd.DataFrame(
        {
            "item_id": "A",
            "timestamp": history_dates,
            "target": targets.to_numpy(),
            "是否雨雪天气": weather.to_numpy(),
        }
    )
    data = history.copy()
    data.loc[len(data)] = {
        "item_id": "A",
        "timestamp": pd.Timestamp("2026-07-02"),
        "target": float("nan"),
        "是否雨雪天气": 1,
    }

    future = generate_baseline_future_forecast(
        data=data,
        freq="D",
        prediction_length=1,
        model=YOY_WEEKDAY_WOW_ORIGINAL_MODEL,
    )

    assert future["forecast_p50"].tolist() == pytest.approx([330.0])


def test_yoy_weekday_wow_original_does_not_replace_on_condition_mismatch_only() -> None:
    history_dates = pd.date_range("2025-06-01", "2026-07-01", freq="D")
    targets = pd.Series(100.0, index=history_dates)
    weather = pd.Series(0, index=history_dates, dtype=int)

    targets.loc[pd.Timestamp("2025-06-26")] = 100.0
    targets.loc[pd.Timestamp("2025-07-03")] = 120.0
    for date in (pd.Timestamp("2025-06-19"), pd.Timestamp("2025-07-17")):
        targets.loc[date - pd.Timedelta(days=7)] = 100.0
        targets.loc[date] = 110.0
        weather.loc[date - pd.Timedelta(days=7)] = 1
        weather.loc[date] = 1
    targets.loc[pd.Timestamp("2026-06-25")] = 300.0
    weather.loc[pd.Timestamp("2026-06-25")] = 1

    history = pd.DataFrame(
        {
            "item_id": "A",
            "timestamp": history_dates,
            "target": targets.to_numpy(),
            "是否雨雪天气": weather.to_numpy(),
        }
    )
    data = history.copy()
    data.loc[len(data)] = {
        "item_id": "A",
        "timestamp": pd.Timestamp("2026-07-02"),
        "target": float("nan"),
        "是否雨雪天气": 1,
    }

    future = generate_baseline_future_forecast(
        data=data,
        freq="D",
        prediction_length=1,
        model=YOY_WEEKDAY_WOW_ORIGINAL_MODEL,
    )

    assert future["forecast_p50"].tolist() == pytest.approx([360.0])


def test_yoy_weekday_dod_applies_prior_year_daily_change() -> None:
    history_dates = pd.date_range("2025-06-01", "2026-07-01", freq="D")
    targets = pd.Series(100.0, index=history_dates)
    targets.loc[pd.Timestamp("2025-07-02")] = 100.0
    targets.loc[pd.Timestamp("2025-07-03")] = 120.0
    targets.loc[pd.Timestamp("2026-07-01")] = 200.0
    history = pd.DataFrame(
        {
            "item_id": "A",
            "timestamp": history_dates,
            "target": targets.to_numpy(),
        }
    )

    future = generate_baseline_future_forecast(
        data=history,
        freq="D",
        prediction_length=1,
        model=YOY_WEEKDAY_DOD_MODEL,
    )

    assert future["forecast_p50"].tolist() == pytest.approx([240.0])


def test_yoy_weekday_dod_rolls_forecast_into_next_day() -> None:
    history_dates = pd.date_range("2025-06-01", "2026-07-01", freq="D")
    targets = pd.Series(100.0, index=history_dates)
    targets.loc[pd.Timestamp("2025-07-02")] = 100.0
    targets.loc[pd.Timestamp("2025-07-03")] = 110.0
    targets.loc[pd.Timestamp("2025-07-04")] = 121.0
    targets.loc[pd.Timestamp("2026-07-01")] = 200.0
    history = pd.DataFrame(
        {
            "item_id": "A",
            "timestamp": history_dates,
            "target": targets.to_numpy(),
        }
    )

    future = generate_baseline_future_forecast(
        data=history,
        freq="D",
        prediction_length=2,
        model=YOY_WEEKDAY_DOD_MODEL,
    ).sort_values("timestamp")

    assert future["forecast_p50"].tolist() == pytest.approx([220.0, 242.0])


def test_yoy_weekday_dod_replaces_volatile_daily_change_even_when_conditions_match() -> None:
    history_dates = pd.date_range("2025-06-01", "2026-07-01", freq="D")
    targets = pd.Series(100.0, index=history_dates)
    targets.loc[pd.Timestamp("2025-07-02")] = 100.0
    targets.loc[pd.Timestamp("2025-07-03")] = 200.0
    for date in (
        pd.Timestamp("2025-06-19"),
        pd.Timestamp("2025-06-26"),
        pd.Timestamp("2025-07-10"),
        pd.Timestamp("2025-07-17"),
    ):
        targets.loc[date - pd.Timedelta(days=1)] = 100.0
        targets.loc[date] = 110.0
    targets.loc[pd.Timestamp("2026-07-01")] = 300.0
    history = pd.DataFrame(
        {
            "item_id": "A",
            "timestamp": history_dates,
            "target": targets.to_numpy(),
        }
    )

    future = generate_baseline_future_forecast(
        data=history,
        freq="D",
        prediction_length=1,
        model=YOY_WEEKDAY_DOD_MODEL,
    )

    assert future["forecast_p50"].tolist() == pytest.approx([330.0])


def test_yoy_weekday_wow_hybrid_selects_seasonal_naive_when_weekly_pattern_wins() -> None:
    dates = pd.date_range("2025-01-01", "2026-07-01", freq="D")
    weekly_pattern = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
    targets = pd.Series(
        [weekly_pattern[index % 7] for index in range(len(dates))],
        index=dates,
    )

    for target_date in pd.date_range("2026-06-25", "2026-07-02", freq="D"):
        aligned = _align_to_prior_year_weekday(target_date)
        targets.loc[aligned - pd.Timedelta(days=7)] = 100.0
        targets.loc[aligned] = 200.0

    history = pd.DataFrame(
        {
            "item_id": "A",
            "timestamp": dates,
            "target": targets.to_numpy(),
        }
    )

    future = generate_baseline_future_forecast(
        data=history,
        freq="D",
        prediction_length=1,
        model=YOY_WEEKDAY_WOW_HYBRID_MODEL,
    )
    selected_weight = _select_yoy_weekday_wow_hybrid_weight(
        history=history,
        context=history,
        prediction_length=1,
    )

    assert selected_weight == 0.0
    assert future["forecast_p50"].tolist() == pytest.approx([weekly_pattern[len(dates) % 7]])


def test_yoy_weekday_wow_hybrid_selects_original_when_yoy_pattern_wins() -> None:
    dates = pd.date_range("2025-01-01", "2026-07-01", freq="D")
    targets = pd.Series(100.0, index=dates)

    for start in pd.date_range("2025-05-01", "2025-05-07", freq="D"):
        for week_index in range(16):
            date = start + pd.Timedelta(days=7 * week_index)
            if date in targets.index:
                targets.loc[date] = 100.0 * (2.0**week_index)
    for start in pd.date_range("2026-06-16", "2026-06-22", freq="D"):
        for week_index in range(3):
            date = start + pd.Timedelta(days=7 * week_index)
            if date in targets.index:
                targets.loc[date] = 100.0 * (2.0**week_index)

    history = pd.DataFrame(
        {
            "item_id": "A",
            "timestamp": dates,
            "target": targets.to_numpy(),
        }
    )

    hybrid = generate_baseline_future_forecast(
        data=history,
        freq="D",
        prediction_length=3,
        model=YOY_WEEKDAY_WOW_HYBRID_MODEL,
    ).sort_values("timestamp")
    original = generate_baseline_future_forecast(
        data=history,
        freq="D",
        prediction_length=3,
        model=YOY_WEEKDAY_WOW_ORIGINAL_MODEL,
    ).sort_values("timestamp")
    selected_weight = _select_yoy_weekday_wow_hybrid_weight(
        history=history,
        context=history,
        prediction_length=3,
    )

    assert selected_weight == 1.0
    assert hybrid["forecast_p50"].tolist() == pytest.approx(original["forecast_p50"].tolist())


def test_yoy_weekday_wow_uses_matching_weather_when_prior_year_conditions_mismatch() -> None:
    history_dates = pd.date_range("2025-06-01", "2026-07-01", freq="D")
    targets = pd.Series(100.0, index=history_dates)
    weather = pd.Series(0, index=history_dates, dtype=int)

    targets.loc[pd.Timestamp("2025-06-26")] = 100.0
    targets.loc[pd.Timestamp("2025-07-03")] = 200.0
    for date in (pd.Timestamp("2025-06-19"), pd.Timestamp("2025-07-17")):
        targets.loc[date - pd.Timedelta(days=7)] = 100.0
        targets.loc[date] = 110.0
        weather.loc[date - pd.Timedelta(days=7)] = 1
        weather.loc[date] = 1
    targets.loc[pd.Timestamp("2025-06-26")] = 100.0
    targets.loc[pd.Timestamp("2025-07-03")] = 200.0
    targets.loc[pd.Timestamp("2026-06-25")] = 300.0
    weather.loc[pd.Timestamp("2026-06-25")] = 1

    history = pd.DataFrame(
        {
            "item_id": "A",
            "timestamp": history_dates,
            "target": targets.to_numpy(),
            "是否雨雪天气": weather.to_numpy(),
        }
    )
    data = history.copy()
    data.loc[len(data)] = {
        "item_id": "A",
        "timestamp": pd.Timestamp("2026-07-02"),
        "target": float("nan"),
        "是否雨雪天气": 1,
    }

    future = generate_baseline_future_forecast(
        data=data,
        freq="D",
        prediction_length=1,
        model=YOY_WEEKDAY_WOW_MODEL,
    )

    assert future["forecast_p50"].tolist() == pytest.approx([330.0])


def test_yoy_weekday_dod_uses_matching_weather_when_prior_year_conditions_mismatch() -> None:
    history_dates = pd.date_range("2025-06-01", "2026-07-01", freq="D")
    targets = pd.Series(100.0, index=history_dates)
    weather = pd.Series(0, index=history_dates, dtype=int)

    targets.loc[pd.Timestamp("2025-07-02")] = 100.0
    targets.loc[pd.Timestamp("2025-07-03")] = 200.0
    for date in (pd.Timestamp("2025-06-19"), pd.Timestamp("2025-07-17")):
        targets.loc[date - pd.Timedelta(days=1)] = 100.0
        targets.loc[date] = 110.0
        weather.loc[date] = 1
    targets.loc[pd.Timestamp("2026-07-01")] = 300.0

    history = pd.DataFrame(
        {
            "item_id": "A",
            "timestamp": history_dates,
            "target": targets.to_numpy(),
            "是否雨雪天气": weather.to_numpy(),
        }
    )
    data = history.copy()
    data.loc[len(data)] = {
        "item_id": "A",
        "timestamp": pd.Timestamp("2026-07-02"),
        "target": float("nan"),
        "是否雨雪天气": 1,
    }

    future = generate_baseline_future_forecast(
        data=data,
        freq="D",
        prediction_length=1,
        model=YOY_WEEKDAY_DOD_MODEL,
    )

    assert future["forecast_p50"].tolist() == pytest.approx([330.0])
