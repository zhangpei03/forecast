import pandas as pd
import pytest

from src.domain.models import ForecastDriverConfig
from src.services.forecast_driver_service import (
    BUILTIN_WEEKDAY_COVARIATE,
    apply_forecast_driver_adjustments,
    apply_growth_rate_adjustments,
    build_future_driver_assumptions,
    build_future_known_covariates,
    known_covariate_columns,
    validate_driver_configs,
)


def test_known_covariates_include_configured_weekday_column() -> None:
    drivers = [
        ForecastDriverConfig(
            name=BUILTIN_WEEKDAY_COVARIATE,
            config_type="covariate",
            column=BUILTIN_WEEKDAY_COVARIATE,
            availability="known_future",
            future_value_strategy="last_value",
        )
    ]

    assert known_covariate_columns(drivers) == [BUILTIN_WEEKDAY_COVARIATE]


def test_build_future_known_covariates_uses_prefilled_weekday_values() -> None:
    data = pd.DataFrame(
        {
            "item_id": ["A", "A", "A", "A"],
            "timestamp": pd.to_datetime(
                ["2024-07-08", "2024-07-09", "2024-07-10", "2024-07-11"]
            ),
            "target": [100.0, 110.0, float("nan"), float("nan")],
            BUILTIN_WEEKDAY_COVARIATE: [1.0, 2.0, 3.0, 4.0],
        }
    )
    drivers = [
        ForecastDriverConfig(
            name=BUILTIN_WEEKDAY_COVARIATE,
            config_type="covariate",
            column=BUILTIN_WEEKDAY_COVARIATE,
            availability="known_future",
            future_value_strategy="last_value",
        )
    ]

    future = build_future_known_covariates(data, drivers, freq="D", prediction_length=2)

    assert future["timestamp"].tolist() == list(pd.date_range("2024-07-10", periods=2, freq="D"))
    assert future[BUILTIN_WEEKDAY_COVARIATE].tolist() == pytest.approx([3.0, 4.0])


def test_build_future_known_covariates_uses_each_series_last_value() -> None:
    data = pd.DataFrame(
        {
            "item_id": ["A", "A", "B", "B"],
            "timestamp": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-01-01", "2024-02-01"]),
            "target": [100.0, 110.0, 200.0, 210.0],
            "workdays": [20.0, 19.0, 21.0, 20.0],
        }
    )
    drivers = [
        ForecastDriverConfig(
            name="工作日数",
            config_type="covariate",
            column="workdays",
            availability="known_future",
            future_value_strategy="last_value",
        )
    ]

    future = build_future_known_covariates(data, drivers, freq="M", prediction_length=2)

    assert future.groupby("item_id")["workdays"].apply(list).to_dict() == {
        "A": [19.0, 19.0],
        "B": [20.0, 20.0],
    }


def test_apply_growth_rate_adjustments_compounds_each_horizon_and_recomputes_errors() -> None:
    predictions = pd.DataFrame(
        {
            "item_id": ["A", "A"],
            "timestamp": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "window_id": ["W1", "W1"],
            "model": ["XGBoost", "XGBoost"],
            "actual": [110.0, 121.0],
            "forecast_mean": [100.0, 100.0],
            "forecast_p10": [90.0, 90.0],
            "forecast_p50": [100.0, 100.0],
            "forecast_p90": [110.0, 110.0],
        }
    )
    drivers = [
        ForecastDriverConfig(
            name="经营增长假设",
            config_type="growth_rate",
            growth_rate=0.10,
        )
    ]

    adjusted = apply_growth_rate_adjustments(predictions, drivers)

    assert adjusted["forecast_p50"].tolist() == pytest.approx([110.0, 121.0])
    assert adjusted["forecast_p10"].tolist() == pytest.approx([99.0, 108.9])
    assert adjusted["error"].tolist() == pytest.approx([0.0, 0.0])
    assert adjusted["error_rate"].tolist() == pytest.approx([0.0, 0.0])


def test_validate_driver_configs_rejects_non_numeric_or_missing_covariates() -> None:
    data = pd.DataFrame(
        {
            "item_id": ["A", "A"],
            "timestamp": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "target": [100.0, 110.0],
            "workdays": [20.0, 19.0],
            "region": ["CN", "CN"],
        }
    )
    drivers = [
        ForecastDriverConfig(
            name="区域",
            config_type="covariate",
            column="region",
            availability="known_future",
            future_value_strategy="last_value",
        ),
        ForecastDriverConfig(
            name="缺失字段",
            config_type="covariate",
            column="missing_column",
            availability="known_future",
            future_value_strategy="last_value",
        ),
    ]

    errors = validate_driver_configs(drivers, data)

    assert any("区域" in error and "数值" in error for error in errors)
    assert any("missing_column" in error for error in errors)


def test_calendar_factor_adjusts_only_configured_months() -> None:
    predictions = pd.DataFrame(
        {
            "item_id": ["A", "A", "A"],
            "timestamp": pd.to_datetime(["2025-04-01", "2025-05-01", "2025-10-01"]),
            "window_id": ["FUTURE"] * 3,
            "model": ["XGBoost"] * 3,
            "actual": [pd.NA] * 3,
            "forecast_mean": [100.0] * 3,
            "forecast_p10": [90.0] * 3,
            "forecast_p50": [100.0] * 3,
            "forecast_p90": [110.0] * 3,
        }
    )
    drivers = [
        ForecastDriverConfig(
            name="五一国庆",
            config_type="calendar_factor",
            impact_months=[5, 10],
            impact_rate=0.12,
        )
    ]

    adjusted = apply_forecast_driver_adjustments(predictions, drivers)

    assert adjusted["forecast_p50"].tolist() == pytest.approx([100.0, 112.0, 112.0])


def test_build_future_driver_assumptions_outputs_missing_covariate_scenarios() -> None:
    data = pd.DataFrame(
        {
            "item_id": ["A", "A", "B", "B"],
            "timestamp": pd.to_datetime(["2025-01-01", "2025-02-01", "2025-01-01", "2025-02-01"]),
            "target": [100.0, 110.0, 200.0, 210.0],
        }
    )
    drivers = [
        ForecastDriverConfig(
            name="计划订单数",
            config_type="scenario_covariate",
            base_value=100.0,
            scenario_growth_rate=0.05,
            effect_rate=0.8,
        )
    ]

    assumptions = build_future_driver_assumptions(data, drivers, freq="M", prediction_length=2)

    assert assumptions.groupby("item_id")["assumption_value"].apply(list).to_dict() == {
        "A": pytest.approx([105.0, 110.25]),
        "B": pytest.approx([105.0, 110.25]),
    }
    assert set(assumptions["adjustment_mode"]) == {"手工影响系数"}


def test_future_driver_assumptions_start_after_last_actual_not_prefilled_row() -> None:
    data = pd.DataFrame(
        {
            "item_id": ["A", "A", "A"],
            "timestamp": pd.to_datetime(["2025-01-01", "2025-02-01", "2025-03-01"]),
            "target": [100.0, 110.0, None],
        }
    )
    drivers = [
        ForecastDriverConfig(
            name="计划订单数",
            config_type="scenario_covariate",
            base_value=100.0,
            scenario_growth_rate=0.05,
            effect_rate=0.8,
        )
    ]

    assumptions = build_future_driver_assumptions(
        data,
        drivers,
        freq="M",
        prediction_length=1,
    )

    assert assumptions["timestamp"].tolist() == [pd.Timestamp("2025-03-01")]


def test_build_future_known_covariates_uses_prefilled_future_rows() -> None:
    """当 Excel 中存在 target=NaN 的未来行时，协变量值应直接使用预填值。"""
    data = pd.DataFrame(
        {
            "item_id": ["A", "A", "A", "A", "A"],
            "timestamp": pd.to_datetime(
                ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01"]
            ),
            "target": [100.0, 110.0, 120.0, float("nan"), float("nan")],
            "holiday": [0.0, 0.0, 0.0, 1.0, 0.0],
        }
    )
    drivers = [
        ForecastDriverConfig(
            name="节假日",
            config_type="covariate",
            column="holiday",
            availability="known_future",
            future_value_strategy="last_value",
        )
    ]
    result = build_future_known_covariates(data, drivers, freq="M", prediction_length=2)
    assert len(result) == 2
    holidays = result.sort_values("timestamp")["holiday"].tolist()
    assert holidays == pytest.approx([1.0, 0.0])


def test_build_future_known_covariates_fallback_when_no_prefill() -> None:
    """没有预填未来行时退回 last_value / mean_value 常数策略。"""
    data = pd.DataFrame(
        {
            "item_id": ["A", "A", "A"],
            "timestamp": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
            "target": [100.0, 110.0, 120.0],
            "holiday": [0.0, 1.0, 0.0],
        }
    )
    drivers_last = [
        ForecastDriverConfig(
            name="节假日",
            config_type="covariate",
            column="holiday",
            availability="known_future",
            future_value_strategy="last_value",
        )
    ]
    drivers_mean = [
        ForecastDriverConfig(
            name="节假日",
            config_type="covariate",
            column="holiday",
            availability="known_future",
            future_value_strategy="mean_value",
        )
    ]
    result_last = build_future_known_covariates(data, drivers_last, freq="M", prediction_length=2)
    result_mean = build_future_known_covariates(data, drivers_mean, freq="M", prediction_length=2)
    assert result_last["holiday"].tolist() == pytest.approx([0.0, 0.0])
    assert result_mean["holiday"].tolist() == pytest.approx([1 / 3, 1 / 3])


def test_build_future_known_covariates_applies_coeff() -> None:
    data = pd.DataFrame(
        {
            "item_id": ["A", "A", "A"],
            "timestamp": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
            "target": [100.0, 110.0, 120.0],
            "holiday": [0.0, 1.0, 0.0],
        }
    )
    # last_value with +20% coeff
    drivers_coeff = [
        ForecastDriverConfig(
            name="节假日",
            config_type="covariate",
            column="holiday",
            availability="known_future",
            future_value_strategy="last_value",
            future_value_coeff=0.2,
        )
    ]
    result = build_future_known_covariates(data, drivers_coeff, freq="M", prediction_length=2)
    # last_value = 0.0, with +20% = 0.0 * 1.2 = 0.0
    assert result["holiday"].tolist() == pytest.approx([0.0, 0.0])

    # mean_value with +10% coeff
    drivers_mean_coeff = [
        ForecastDriverConfig(
            name="节假日",
            config_type="covariate",
            column="holiday",
            availability="known_future",
            future_value_strategy="mean_value",
            future_value_coeff=0.1,
        )
    ]
    result2 = build_future_known_covariates(data, drivers_mean_coeff, freq="M", prediction_length=2)
    # mean = (0+1+0)/3 = 1/3, with +10% = 1/3 * 1.1
    expected = (1 / 3) * 1.1
    assert result2["holiday"].tolist() == pytest.approx([expected, expected])
