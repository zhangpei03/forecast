from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


def export_evaluation_workbook(
    *,
    output_dir: Path,
    experiment_name: str,
    conclusion: str,
    leaderboard: pd.DataFrame,
    aggregate_metrics: pd.DataFrame,
    series_metrics: pd.DataFrame,
    backtest_predictions: pd.DataFrame,
    future_forecast: pd.DataFrame,
    future_driver_assumptions: pd.DataFrame,
    quality_report: pd.DataFrame,
    config: dict,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(char if char.isalnum() or char in "-_" else "_" for char in experiment_name)
    output_path = output_dir / f"{safe_name}_预测评测_{datetime.now():%Y%m%d_%H%M}.xlsx"

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pd.DataFrame([{"业务结论": conclusion}]).to_excel(
            writer, sheet_name="评测结论", index=False
        )
        leaderboard.to_excel(writer, sheet_name="模型排行榜", index=False)
        aggregate_metrics.to_excel(writer, sheet_name="回测汇总", index=False)
        series_metrics.to_excel(writer, sheet_name="序列评测", index=False)
        backtest_predictions.to_excel(writer, sheet_name="期间偏差明细", index=False)
        future_forecast.to_excel(writer, sheet_name="未来预测", index=False)
        future_driver_assumptions.to_excel(writer, sheet_name="未来驱动假设", index=False)
        quality_report.to_excel(writer, sheet_name="数据质量", index=False)
        pd.DataFrame([config]).to_excel(writer, sheet_name="实验配置", index=False)

        for worksheet in writer.book.worksheets:
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for column_cells in worksheet.columns:
                max_length = max(len(str(cell.value or "")) for cell in column_cells)
                worksheet.column_dimensions[column_cells[0].column_letter].width = min(
                    max_length + 2, 42
                )

    return output_path
