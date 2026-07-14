from __future__ import annotations

import pandas as pd
import pytest

from src.core.constants import BASELINE_MODEL_NAMES
from src.services.baseline_service import (
    DOD2_MODEL,
    YOY_WEEKDAY_DOD_ANCHOR_MODEL,
    YOY_WEEKDAY_DOD_ANCHOR_V2_MODEL,
    YOY_WEEKDAY_DOD_MODEL,
    YOY_WEEKDAY_WOW_MODEL,
    _align_daily_ratio_date,
    _align_to_prior_year_weekday,
    generate_baseline_backtest_predictions,
    generate_baseline_future_forecast,
)


def test_generate_baseline_backtest_predictions_for_monthly_series() -> None:
    dates = pd.date_range("2022-01-01", periods=30, freq="MS")
    data = pd.DataFrame(
        {"item_id": ["A"] * 30, "timestamp": dates, "target": [float(i + 1) for i in range(30)]}
    )

    predictions = generate_baseline_backtest_predictions(
        data=data, freq="M", prediction_length=3, num_windows=2
    )

    assert set(predictions["model"]) == {"Last Value", "Seasonal Naive", "Rolling Mean"}


def test_daily_baseline_catalog_keeps_only_the_renamed_weekday_wow_model() -> None:
    dates = pd.date_range("2024-01-01", periods=400, freq="D")
    data = pd.DataFrame(
        {"item_id": ["A"] * len(dates), "timestamp": dates, "target": [float(i + 1) for i in range(len(dates))]}
    )

    predictions = generate_baseline_backtest_predictions(
        data=data, freq="D", prediction_length=30, num_windows=1
    )

    models = set(predictions["model"])
    assert YOY_WEEKDAY_WOW_MODEL in models
    assert YOY_WEEKDAY_DOD_MODEL in models
    assert YOY_WEEKDAY_DOD_ANCHOR_MODEL in models
    assert YOY_WEEKDAY_DOD_ANCHOR_V2_MODEL in models
    assert DOD2_MODEL in models
    assert "YoY Weekday WoW Original" not in models
    assert "YoY Weekday WoW Hybrid" not in models
    assert "YoY Weekday WoW Original" not in BASELINE_MODEL_NAMES
    assert "YoY Weekday WoW Hybrid" not in BASELINE_MODEL_NAMES


def test_legacy_daily_model_names_are_normalized_to_standardized_labels() -> None:
    dates = pd.date_range("2025-01-01", periods=400, freq="D")
    data = pd.DataFrame(
        {"item_id": "A", "timestamp": dates, "target": [100.0] * len(dates)}
    )

    predictions = generate_baseline_backtest_predictions(
        data=data,
        freq="D",
        prediction_length=1,
        num_windows=1,
        models=["DoD2"],
    )

    assert set(predictions["model"]) == {DOD2_MODEL}


def test_selected_baseline_runs_only_the_requested_model() -> None:
    dates = pd.date_range("2022-01-01", periods=30, freq="MS")
    data = pd.DataFrame(
        {"item_id": ["A"] * 30, "timestamp": dates, "target": [float(i + 1) for i in range(30)]}
    )

    predictions = generate_baseline_backtest_predictions(
        data=data, freq="M", prediction_length=3, num_windows=1, models=["Rolling Mean"]
    )

    assert set(predictions["model"]) == {"Rolling Mean"}


def test_yoy_weekday_wow_keeps_the_raw_ratio_when_it_differs_from_nearby_values() -> None:
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

    data = pd.DataFrame(
        {
            "item_id": "A",
            "timestamp": history_dates,
            "target": targets.to_numpy(),
            "是否雨雪天气": weather.to_numpy(),
        }
    )
    data.loc[len(data)] = {
        "item_id": "A",
        "timestamp": pd.Timestamp("2026-07-02"),
        "target": float("nan"),
        "是否雨雪天气": 1,
    }

    future = generate_baseline_future_forecast(
        data=data, freq="D", prediction_length=1, model=YOY_WEEKDAY_WOW_MODEL
    )

    row = future.iloc[0]
    assert row["forecast_p50"] == pytest.approx(600.0)
    assert row["预测依据"] == "前一周值 × 去年周环比"
    assert row["预测基准日期"] == pd.Timestamp("2026-06-25")
    assert row["预测基准值"] == pytest.approx(300.0)
    assert row["去年环比日期"] == pd.Timestamp("2025-07-03")
    assert row["去年环比值"] == pytest.approx(200.0)
    assert row["去年环比对比日期"] == pd.Timestamp("2025-06-26")
    assert row["去年环比对比值"] == pytest.approx(100.0)
    assert row["实际采用环比"] == pytest.approx(1.0)
    assert row["环比来源"] == "去年对齐日周环比"


def test_yoy_weekday_dod_records_the_values_used_for_each_forecast() -> None:
    history_dates = pd.date_range("2025-06-01", "2026-07-01", freq="D")
    targets = pd.Series(100.0, index=history_dates)
    targets.loc[pd.Timestamp("2025-07-02")] = 100.0
    targets.loc[pd.Timestamp("2025-07-03")] = 120.0
    targets.loc[pd.Timestamp("2026-07-01")] = 200.0
    history = pd.DataFrame(
        {"item_id": "A", "timestamp": history_dates, "target": targets.to_numpy()}
    )

    future = generate_baseline_future_forecast(
        data=history, freq="D", prediction_length=1, model=YOY_WEEKDAY_DOD_MODEL
    )

    row = future.iloc[0]
    assert row["forecast_p50"] == pytest.approx(240.0)
    assert row["预测依据"] == "前一天值 × 去年日环比"
    assert row["预测基准日期"] == pd.Timestamp("2026-07-01")
    assert row["预测基准值"] == pytest.approx(200.0)
    assert row["去年环比日期"] == pd.Timestamp("2025-07-03")
    assert row["去年环比值"] == pytest.approx(120.0)
    assert row["去年环比对比日期"] == pd.Timestamp("2025-07-02")
    assert row["去年环比对比值"] == pytest.approx(100.0)
    assert row["实际采用环比"] == pytest.approx(0.2)
    assert row["环比来源"] == "去年对齐日日环比"


def test_dod2_uses_weekday_alignment_and_ignores_special_holiday() -> None:
    history_dates = pd.date_range("2025-05-01", "2026-06-18", freq="D")
    targets = pd.Series(100.0, index=history_dates)
    targets.loc[pd.Timestamp("2025-05-30")] = 80.0
    targets.loc[pd.Timestamp("2025-05-31")] = 400.0
    targets.loc[pd.Timestamp("2025-06-19")] = 100.0
    targets.loc[pd.Timestamp("2025-06-20")] = 150.0
    targets.loc[pd.Timestamp("2026-06-18")] = 200.0
    data = pd.DataFrame(
        {
            "item_id": "A",
            "timestamp": history_dates,
            "target": targets.to_numpy(),
            "特殊假期": "None",
        }
    )
    data.loc[len(data)] = {
        "item_id": "A",
        "timestamp": pd.Timestamp("2026-06-19"),
        "target": float("nan"),
        "特殊假期": "D1",
    }

    future = generate_baseline_future_forecast(
        data=data, freq="D", prediction_length=1, model=DOD2_MODEL
    )

    row = future.iloc[0]
    assert row["forecast_p50"] == pytest.approx(300.0)
    assert row["去年环比日期"] == pd.Timestamp("2025-06-20")
    assert row["去年环比对比日期"] == pd.Timestamp("2025-06-19")


def test_yoy_weekday_dod_anchor_blends_holiday_ratio_with_annual_level() -> None:
    history_dates = pd.date_range("2025-05-01", "2026-06-18", freq="D")
    targets = pd.Series(100.0, index=history_dates)
    targets.loc[pd.Timestamp("2025-05-30")] = 80.0
    targets.loc[pd.Timestamp("2025-05-31")] = 400.0
    targets.loc[pd.Timestamp("2026-06-18")] = 200.0
    data = pd.DataFrame(
        {
            "item_id": "A",
            "timestamp": history_dates,
            "target": targets.to_numpy(),
            "特殊假期": "None",
        }
    )
    data.loc[data["timestamp"].eq(pd.Timestamp("2025-05-31")), "特殊假期"] = "D1"
    data.loc[len(data)] = {
        "item_id": "A",
        "timestamp": pd.Timestamp("2026-06-19"),
        "target": float("nan"),
        "特殊假期": "D1",
    }

    future = generate_baseline_future_forecast(
        data=data, freq="D", prediction_length=1, model=YOY_WEEKDAY_DOD_ANCHOR_MODEL
    )

    row = future.iloc[0]
    assert row["forecast_p50"] == pytest.approx(760.0)
    assert row["预测依据"] == "日环比递推 60% + 年度水平锚点 40%"
    assert row["去年环比日期"] == pd.Timestamp("2025-05-31")
    assert row["去年环比对比日期"] == pd.Timestamp("2025-05-30")


def test_yoy_weekday_dod_anchor_v2_uses_holiday_alignment_and_adaptive_blend() -> None:
    history_dates = pd.date_range("2025-05-01", "2026-06-18", freq="D")
    targets = pd.Series(100.0, index=history_dates)
    targets.loc[pd.Timestamp("2025-05-30")] = 80.0
    targets.loc[pd.Timestamp("2025-05-31")] = 400.0
    targets.loc[pd.Timestamp("2026-06-18")] = 200.0
    data = pd.DataFrame(
        {
            "item_id": "A",
            "timestamp": history_dates,
            "target": targets.to_numpy(),
            "特殊假期": "None",
        }
    )
    data.loc[data["timestamp"].eq(pd.Timestamp("2025-05-31")), "特殊假期"] = "D1"
    data.loc[len(data)] = {
        "item_id": "A",
        "timestamp": pd.Timestamp("2026-06-19"),
        "target": float("nan"),
        "特殊假期": "D1",
    }

    future = generate_baseline_future_forecast(
        data=data, freq="D", prediction_length=1, model=YOY_WEEKDAY_DOD_ANCHOR_V2_MODEL
    )

    row = future.iloc[0]
    assert row["forecast_p50"] == pytest.approx(580.0)
    assert row["预测依据"] == "自适应日环比递推 30% + 年度水平锚点 70%"
    assert row["去年环比日期"] == pd.Timestamp("2025-05-31")
    assert row["去年环比对比日期"] == pd.Timestamp("2025-05-30")


def test_prior_year_alignment_uses_nearest_matching_weekday() -> None:
    aligned = _align_to_prior_year_weekday(pd.Timestamp("2026-07-02"))

    assert aligned == pd.Timestamp("2025-07-03")
    assert aligned.weekday() == pd.Timestamp("2026-07-02").weekday()


def test_daily_ratio_alignment_prioritizes_holiday_code_before_weekday() -> None:
    context = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-09-17", "2025-10-01", "2026-09-25"]),
            "是否节假日": [1, 1, 1],
            "特殊假期": ["Z1", "G1", "Z1"],
        }
    ).set_index("timestamp")

    aligned, source = _align_daily_ratio_date(pd.Timestamp("2026-09-25"), context)

    assert aligned == pd.Timestamp("2024-09-17")
    assert source == "前年同假期日环比"


def test_daily_ratio_alignment_ignores_generic_holiday_flag() -> None:
    context = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2025-05-30", "2025-05-31", "2025-06-20", "2025-06-21", "2026-06-19"]
            ),
            "是否节假日": [0, 1, 0, 1, 1],
            "特殊假期": ["None", "D1", "None", "None", "D1"],
        }
    ).set_index("timestamp")

    aligned, source = _align_daily_ratio_date(pd.Timestamp("2026-06-19"), context)

    assert aligned == pd.Timestamp("2025-05-31")
    assert source == "去年同假期日环比"


def test_daily_ratio_alignment_uses_nearest_prior_year_holiday_with_same_day_number() -> None:
    context = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-10-01", "2025-05-01", "2026-09-25"]),
            "特殊假期": ["G1", "L1", "Z1"],
        }
    ).set_index("timestamp")

    aligned, source = _align_daily_ratio_date(pd.Timestamp("2026-09-25"), context)

    assert aligned == pd.Timestamp("2025-10-01")
    assert source == "去年最近同假期天数日环比"


def test_daily_ratio_alignment_uses_weekday_for_none_holiday_code() -> None:
    target = pd.Timestamp("2026-07-02")
    context = pd.DataFrame(
        {"timestamp": [target], "特殊假期": ["None"]}
    ).set_index("timestamp")

    aligned, source = _align_daily_ratio_date(target, context)

    assert aligned == _align_to_prior_year_weekday(target)
    assert source == "去年对齐日日环比"
