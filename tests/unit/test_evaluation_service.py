import math

import pandas as pd

from src.services.evaluation_service import (
    build_business_conclusion,
    calculate_prediction_metrics,
    choose_best_model,
    classify_stability,
)


def test_calculate_prediction_metrics_handles_zero_actual_and_coverage() -> None:
    predictions = pd.DataFrame(
        {
            "actual": [100.0, 0.0, -50.0],
            "forecast_p50": [110.0, 5.0, -60.0],
            "forecast_p10": [90.0, -10.0, -70.0],
            "forecast_p90": [120.0, 10.0, -40.0],
        }
    )

    metrics = calculate_prediction_metrics(predictions)

    assert metrics.wape == 25 / 150
    assert metrics.mae == 25 / 3
    assert round(metrics.rmse, 4) == round(math.sqrt((100 + 25 + 100) / 3), 4)
    assert metrics.bias_amount == 5
    assert metrics.bias_rate == 5 / 150
    assert metrics.coverage == 1.0


def test_choose_best_model_sorts_by_wape_bias_mae_and_runtime() -> None:
    leaderboard = pd.DataFrame(
        [
            {
                "model": "A",
                "model_type": "AutoGluon",
                "wape": 0.12,
                "bias_rate": 0.02,
                "mae": 10,
                "training_seconds": 8,
            },
            {
                "model": "B",
                "model_type": "AutoGluon",
                "wape": 0.10,
                "bias_rate": -0.03,
                "mae": 12,
                "training_seconds": 9,
            },
            {
                "model": "C",
                "model_type": "AutoGluon",
                "wape": 0.10,
                "bias_rate": 0.01,
                "mae": 12,
                "training_seconds": 11,
            },
            {
                "model": "D",
                "model_type": "AutoGluon",
                "wape": 0.10,
                "bias_rate": 0.01,
                "mae": 11,
                "training_seconds": 20,
            },
        ]
    )

    best = choose_best_model(leaderboard)

    assert best["model"] == "D"


def test_classify_stability_uses_window_win_rate() -> None:
    assert classify_stability(3, 3) == "稳定领先"
    assert classify_stability(2, 3) == "有一定优势"
    assert classify_stability(1, 3) == "不稳定"
    assert classify_stability(1, 1) == "不可判断"


def test_build_business_conclusion_is_metric_based() -> None:
    conclusion = build_business_conclusion(
        best_model="WeightedEnsemble",
        best_wape=0.086,
        best_baseline="Seasonal Naive",
        baseline_wape=0.134,
        improved_windows=3,
        total_windows=3,
        high_risk_series_count=8,
    )

    assert "WeightedEnsemble" in conclusion
    assert "8.6%" in conclusion
    assert "35.8%" in conclusion
    assert "稳定领先" in conclusion
