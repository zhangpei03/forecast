from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.constants import PREDICTION_PROVENANCE_COLUMNS

EXPORT_BACKTEST_WINDOW_IDS = ("W1", "W2", "W3")


def prepare_export_frames(
    *,
    normalized_data: pd.DataFrame,
    backtest_predictions: pd.DataFrame,
    future_forecast: pd.DataFrame,
    best_model: str,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the two user-facing export tables with the original column names.

    The deviation table uses the most recent backtest window of the winning model,
    which is the historical period immediately before the future forecast interval.
    """

    timestamp_column = str(config.get("timestamp_column") or "timestamp")
    target_column = str(config.get("target_column") or "target")
    item_columns = [str(column) for column in config.get("item_columns", [])]
    forecast_column = f"{target_column}预测值"

    dimensions = _dimensions_by_timestamp(normalized_data, item_columns)
    deviation_source = _latest_backtest_window(backtest_predictions, best_model)
    deviation = _with_original_columns(
        deviation_source,
        dimensions,
        item_columns=item_columns,
        timestamp_column=timestamp_column,
    )
    future = _with_original_columns(
        future_forecast,
        dimensions,
        item_columns=item_columns,
        timestamp_column=timestamp_column,
    )

    deviation_columns = [
        *item_columns,
        timestamp_column,
        "预测算法",
        target_column,
        forecast_column,
        "误差率",
        *PREDICTION_PROVENANCE_COLUMNS,
    ]
    future_columns = [
        *item_columns,
        timestamp_column,
        "预测算法",
        forecast_column,
        *PREDICTION_PROVENANCE_COLUMNS,
    ]
    deviation_export = pd.DataFrame(
        {
            **{column: deviation.get(column, pd.NA) for column in item_columns},
            timestamp_column: deviation.get(timestamp_column, pd.NaT),
            "预测算法": deviation.get("model", best_model),
            target_column: deviation.get("actual", pd.NA),
            forecast_column: _forecast_values(deviation),
            "误差率": deviation.get("error_rate", pd.NA),
            **{column: deviation.get(column, pd.NA) for column in PREDICTION_PROVENANCE_COLUMNS},
        }
    )
    future_export = pd.DataFrame(
        {
            **{column: future.get(column, pd.NA) for column in item_columns},
            timestamp_column: future.get(timestamp_column, pd.NaT),
            "预测算法": future.get("model", best_model),
            forecast_column: _forecast_values(future),
            **{column: future.get(column, pd.NA) for column in PREDICTION_PROVENANCE_COLUMNS},
        }
    )
    return (
        deviation_export.loc[:, deviation_columns],
        future_export.loc[:, future_columns],
    )


def prepare_backtest_export_frames(
    *,
    normalized_data: pd.DataFrame,
    backtest_predictions: pd.DataFrame,
    future_forecast: pd.DataFrame,
    best_model: str,
    config: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Build W1-W3 backtest exports plus the future forecast table.

    Every backtest sheet retains the current deviation-comparison layout.  Empty
    windows are included with headers so the workbook layout stays consistent
    when an experiment has fewer valid backtest windows.
    """

    timestamp_column = str(config.get("timestamp_column") or "timestamp")
    target_column = str(config.get("target_column") or "target")
    item_columns = [str(column) for column in config.get("item_columns", [])]
    forecast_column = f"{target_column}预测值"
    deviation_columns = [
        *item_columns,
        timestamp_column,
        "预测算法",
        target_column,
        forecast_column,
        "误差率",
        *PREDICTION_PROVENANCE_COLUMNS,
    ]
    future_columns = [
        *item_columns,
        timestamp_column,
        "预测算法",
        forecast_column,
        *PREDICTION_PROVENANCE_COLUMNS,
    ]

    dimensions = _dimensions_by_timestamp(normalized_data, item_columns)
    selected = backtest_predictions[
        backtest_predictions.get("model", pd.Series(dtype=str)).eq(best_model)
    ].copy()
    if not selected.empty:
        selected["timestamp"] = pd.to_datetime(selected["timestamp"])

    window_exports: dict[str, pd.DataFrame] = {}
    for window_id in EXPORT_BACKTEST_WINDOW_IDS:
        source = selected[selected.get("window_id", pd.Series(dtype=str)).eq(window_id)].copy()
        deviation = _with_original_columns(
            source,
            dimensions,
            item_columns=item_columns,
            timestamp_column=timestamp_column,
        )
        window_exports[window_id] = pd.DataFrame(
            {
                **{
                    column: _column_or_default(deviation, column, pd.NA)
                    for column in item_columns
                },
                timestamp_column: _column_or_default(deviation, timestamp_column, pd.NaT),
                "预测算法": _column_or_default(deviation, "model", best_model),
                target_column: _column_or_default(deviation, "actual", pd.NA),
                forecast_column: _forecast_values(deviation),
                "误差率": _column_or_default(deviation, "error_rate", pd.NA),
                **{
                    column: _column_or_default(deviation, column, pd.NA)
                    for column in PREDICTION_PROVENANCE_COLUMNS
                },
            }
        ).reindex(columns=deviation_columns)

    future = _with_original_columns(
        future_forecast,
        dimensions,
        item_columns=item_columns,
        timestamp_column=timestamp_column,
    )
    future_export = pd.DataFrame(
        {
            **{column: _column_or_default(future, column, pd.NA) for column in item_columns},
            timestamp_column: _column_or_default(future, timestamp_column, pd.NaT),
            "预测算法": _column_or_default(future, "model", best_model),
            forecast_column: _forecast_values(future),
            **{
                column: _column_or_default(future, column, pd.NA)
                for column in PREDICTION_PROVENANCE_COLUMNS
            },
        }
    ).reindex(columns=future_columns)
    return window_exports, future_export


def export_evaluation_workbook(
    *,
    output_dir: Path,
    experiment_name: str,
    normalized_data: pd.DataFrame,
    backtest_predictions: pd.DataFrame,
    future_forecast: pd.DataFrame,
    best_model: str,
    config: dict[str, Any],
) -> Path:
    """Export W1-W3 deviation comparisons and the future forecast table."""

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(char if char.isalnum() or char in "-_" else "_" for char in experiment_name)
    output_path = output_dir / f"{safe_name}_预测结果_{datetime.now():%Y%m%d_%H%M}.xlsx"
    window_exports, future = prepare_backtest_export_frames(
        normalized_data=normalized_data,
        backtest_predictions=backtest_predictions,
        future_forecast=future_forecast,
        best_model=best_model,
        config=config,
    )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for window_id, deviation in window_exports.items():
            deviation.to_excel(writer, sheet_name=window_id, index=False)
        future.to_excel(writer, sheet_name="未来预测", index=False)

        for worksheet in writer.book.worksheets:
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for column_cells in worksheet.columns:
                header = str(column_cells[0].value or "")
                if header.endswith("误差率") or header in {"误差率", "实际采用环比"}:
                    for cell in column_cells[1:]:
                        cell.number_format = "0.00%"
                elif header.endswith("日期"):
                    for cell in column_cells[1:]:
                        cell.number_format = "yyyy-mm-dd"
                elif header.endswith("预测值") or header == str(config.get("target_column") or ""):
                    for cell in column_cells[1:]:
                        cell.number_format = "#,##0.00"
                elif header in {"预测基准值", "去年环比值", "去年环比对比值"}:
                    for cell in column_cells[1:]:
                        cell.number_format = "#,##0.00"
                max_length = max(len(str(cell.value or "")) for cell in column_cells)
                worksheet.column_dimensions[column_cells[0].column_letter].width = min(
                    max_length + 2, 42
                )

    return output_path


def _dimensions_by_timestamp(data: pd.DataFrame, item_columns: list[str]) -> pd.DataFrame:
    columns = ["item_id", "timestamp", *item_columns]
    available = [column for column in columns if column in data.columns]
    dimensions = data.loc[:, available].copy()
    dimensions["timestamp"] = pd.to_datetime(dimensions["timestamp"])
    return dimensions.drop_duplicates(["item_id", "timestamp"], keep="last")


def _latest_backtest_window(predictions: pd.DataFrame, best_model: str) -> pd.DataFrame:
    selected = predictions[predictions.get("model", pd.Series(dtype=str)).eq(best_model)].copy()
    if selected.empty:
        return selected
    selected["timestamp"] = pd.to_datetime(selected["timestamp"])
    latest_timestamp = selected["timestamp"].max()
    latest_windows = selected.loc[selected["timestamp"].eq(latest_timestamp), "window_id"].dropna()
    if latest_windows.empty:
        return selected.loc[selected["timestamp"].eq(latest_timestamp)]
    return selected[selected["window_id"].eq(latest_windows.iloc[0])]


def _with_original_columns(
    predictions: pd.DataFrame,
    dimensions: pd.DataFrame,
    *,
    item_columns: list[str],
    timestamp_column: str,
) -> pd.DataFrame:
    result = predictions.copy()
    if result.empty:
        return result
    result["timestamp"] = pd.to_datetime(result["timestamp"])
    result = result.merge(dimensions, on=["item_id", "timestamp"], how="left")
    return result.rename(columns={"timestamp": timestamp_column})


def _forecast_values(data: pd.DataFrame) -> pd.Series:
    if "forecast_p50" in data:
        return data["forecast_p50"]
    return _column_or_default(data, "forecast_mean", pd.NA)


def _column_or_default(data: pd.DataFrame, column: str, default: object) -> pd.Series:
    if column in data:
        return data[column]
    return pd.Series(default, index=data.index)
