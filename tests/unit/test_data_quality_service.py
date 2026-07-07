import pandas as pd

from src.services.data_quality_service import (
    detect_missing_periods,
    detect_series_frequency,
    estimate_supported_backtest_windows,
    profile_normalized_data,
)


def test_detect_series_frequency_identifies_monthly_data() -> None:
    timestamps = pd.date_range("2024-01-01", periods=6, freq="MS")

    assert detect_series_frequency(timestamps) == "M"


def test_detect_missing_periods_reports_gap_per_series() -> None:
    data = pd.DataFrame(
        {
            "item_id": ["A", "A", "A"],
            "timestamp": pd.to_datetime(["2024-01-01", "2024-03-01", "2024-04-01"]),
            "target": [1.0, 2.0, 3.0],
        }
    )

    gaps = detect_missing_periods(data, "M")

    assert gaps == {"A": ["2024-02"]}


def test_profile_normalized_data_counts_blocking_and_warning_issues() -> None:
    data = pd.DataFrame(
        {
            "item_id": ["A", "A", "A", "A"],
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-03-01", "2024-04-01"]),
            "target": [1.0, None, 0.0, 0.0],
        }
    )

    profile = profile_normalized_data(data, freq="M", prediction_length=3)

    assert profile.blocking_issue_count >= 2
    assert profile.warning_count >= 1
    assert profile.row_count == 4


def test_future_covariate_rows_do_not_extend_history_or_backtest_capacity() -> None:
    timestamps = pd.date_range("2026-01-01", periods=10, freq="D")
    data = pd.DataFrame(
        {
            "item_id": ["A"] * 10,
            "timestamp": timestamps,
            "target": [1.0] * 7 + [None] * 3,
            "weather": [0] * 7 + [1, 0, 1],
        }
    )

    profile = profile_normalized_data(data, freq="D", prediction_length=2)
    windows = estimate_supported_backtest_windows(
        data,
        prediction_length=2,
        requested_windows=5,
    )

    assert profile.data_end == "2026-01-07"
    assert profile.average_series_length == 7.0
    assert windows == 2
