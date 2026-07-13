from __future__ import annotations

import pandas as pd

from src.core.constants import PREDICTION_PROVENANCE_COLUMNS
from src.services.export_service import prepare_export_frames


def test_prepare_export_frames_uses_original_column_names_and_latest_window() -> None:
    normalized = pd.DataFrame(
        {
            "item_id": ["north", "north", "north"],
            "timestamp": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            "区域": ["北区", "北区", "北区"],
            "日期": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "GMV": [100.0, 120.0, pd.NA],
        }
    )
    backtest = pd.DataFrame(
        {
            "item_id": ["north", "north", "north"],
            "timestamp": pd.to_datetime(["2025-12-01", "2025-12-02", "2026-01-02"]),
            "window_id": ["W2", "W2", "W1"],
            "model": ["XGBoost", "XGBoost", "XGBoost"],
            "actual": [80.0, 90.0, 120.0],
            "forecast_p50": [82.0, 95.0, 118.0],
            "error_rate": [0.025, 0.056, -0.017],
            "预测依据": ["前一周值 × 去年周环比"] * 3,
            "实际采用环比": [0.1] * 3,
        }
    )
    future = pd.DataFrame(
        {
            "item_id": ["north"],
            "timestamp": pd.to_datetime(["2026-01-03"]),
            "forecast_p50": [125.0],
            "预测依据": ["前一周值 × 去年周环比"],
            "实际采用环比": [0.1],
        }
    )

    deviation, future_export = prepare_export_frames(
        normalized_data=normalized,
        backtest_predictions=backtest,
        future_forecast=future,
        best_model="XGBoost",
        config={"timestamp_column": "日期", "target_column": "GMV", "item_columns": ["区域"]},
    )

    assert deviation.columns.tolist() == [
        "区域",
        "日期",
        "预测算法",
        "GMV",
        "GMV预测值",
        "误差率",
        *PREDICTION_PROVENANCE_COLUMNS,
    ]
    assert deviation.shape[0] == 1
    assert deviation.iloc[0][["区域", "日期", "预测算法", "GMV", "GMV预测值", "误差率"]].to_dict() == {
        "区域": "北区",
        "日期": pd.Timestamp("2026-01-02"),
        "预测算法": "XGBoost",
        "GMV": 120.0,
        "GMV预测值": 118.0,
        "误差率": -0.017,
    }
    assert future_export.loc[0, ["区域", "日期", "预测算法", "GMV预测值"]].to_dict() == {
        "区域": "北区",
        "日期": pd.Timestamp("2026-01-03"),
        "预测算法": "XGBoost",
        "GMV预测值": 125.0,
    }
    assert list(deviation.columns[-len(PREDICTION_PROVENANCE_COLUMNS) :]) == list(
        PREDICTION_PROVENANCE_COLUMNS
    )
    assert list(future_export.columns[-len(PREDICTION_PROVENANCE_COLUMNS) :]) == list(
        PREDICTION_PROVENANCE_COLUMNS
    )
    assert deviation.loc[0, "预测依据"] == "前一周值 × 去年周环比"
    assert deviation.loc[0, "实际采用环比"] == 0.1
    assert future_export.loc[0, "预测依据"] == "前一周值 × 去年周环比"
    assert future_export.loc[0, "实际采用环比"] == 0.1
