import pandas as pd

from src.services.chart_service import build_actual_vs_forecast_figure, build_trend_figure


def test_build_trend_figure_recomputes_error_rate_for_aggregate_view() -> None:
    actuals = pd.DataFrame(
        {
            "item_id": ["A", "B"],
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "target": [100.0, 200.0],
        }
    )
    backtest = pd.DataFrame(
        {
            "item_id": ["A", "B"],
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "model": ["AutoGluon", "AutoGluon"],
            "actual": [100.0, 200.0],
            "forecast_p50": [110.0, 210.0],
            "forecast_p10": [90.0, 190.0],
            "forecast_p90": [120.0, 220.0],
            "error": [10.0, 10.0],
        }
    )
    future = pd.DataFrame(
        columns=[
            "item_id",
            "timestamp",
            "model",
            "forecast_p50",
            "forecast_p10",
            "forecast_p90",
        ]
    )

    figure = build_trend_figure(
        actuals=actuals,
        backtest_predictions=backtest,
        future_forecast=future,
        model="AutoGluon",
    )

    assert len(figure.data) >= 2


def test_build_actual_vs_forecast_figure_supports_aggregate_view() -> None:
    predictions = pd.DataFrame(
        {
            "item_id": ["A", "B"],
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "model": ["AutoARIMA", "AutoARIMA"],
            "actual": [100.0, 200.0],
            "forecast_mean": [110.0, 210.0],
            "forecast_p50": [110.0, 210.0],
            "forecast_p10": [90.0, 190.0],
            "forecast_p90": [120.0, 220.0],
            "error": [10.0, 10.0],
        }
    )

    figure = build_actual_vs_forecast_figure(predictions, model="AutoARIMA")

    assert len(figure.data) == 2
    assert list(figure.data[1].text) == ["全部序列 · 2024-01-01"]
