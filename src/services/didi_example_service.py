from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def build_synthetic_didi_finance_data() -> pd.DataFrame:
    """Build deterministic synthetic mobility-finance data for the forecasting walkthrough."""

    periods = pd.date_range("2022-01-01", periods=48, freq="MS")
    business_lines = {
        "网约车": {"orders": 980, "take_rate": 10.8, "cost_ratio": 0.70, "cities": 295},
        "顺风车": {"orders": 310, "take_rate": 8.9, "cost_ratio": 0.61, "cities": 210},
        "企业出行": {"orders": 165, "take_rate": 12.4, "cost_ratio": 0.65, "cities": 78},
        "两轮车": {"orders": 440, "take_rate": 4.2, "cost_ratio": 0.58, "cities": 175},
    }
    rng = np.random.default_rng(20260622)
    rows: list[dict[str, object]] = []

    for business_index, (business_line, parameters) in enumerate(business_lines.items()):
        for period_index, period in enumerate(periods):
            spring_festival = int(period.month == 2)
            workdays = (
                len(pd.bdate_range(period, period + pd.offsets.MonthEnd(0))) - spring_festival * 4
            )
            fuel_index = (
                100 + period_index * 0.18 + 4.5 * np.sin(period_index / 5) + rng.normal(0, 0.8)
            )
            seasonal = 1 + 0.08 * np.sin((period.month - 1) / 12 * 2 * np.pi)
            festival_effect = 0.83 if spring_festival else 1.0
            plan_orders = (
                parameters["orders"]
                * (1 + period_index * 0.004)
                * seasonal
                * festival_effect
                * (1 + rng.normal(0, 0.012))
            )
            actual_orders = plan_orders * (1 + rng.normal(0, 0.018))
            city_coverage = parameters["cities"] + period_index // 12 * (2 + business_index)
            revenue = actual_orders * 10_000 * parameters["take_rate"] * (1 + rng.normal(0, 0.015))
            operating_cost = (
                revenue
                * parameters["cost_ratio"]
                * (1 + (fuel_index - 100) * 0.002 + rng.normal(0, 0.012))
            )
            for account, amount in (("订单收入", revenue), ("运营成本", operating_cost)):
                rows.append(
                    {
                        "期间": period.strftime("%Y-%m"),
                        "业务线": business_line,
                        "财务科目": account,
                        "实际金额（元）": round(float(amount), 2),
                        "计划订单量（万单）": round(float(plan_orders), 2),
                        "工作日数": int(workdays),
                        "燃油价格指数": round(float(fuel_index), 2),
                        "城市覆盖数": int(city_coverage),
                        "是否春节": spring_festival,
                    }
                )
    return pd.DataFrame(rows)


def write_example_workbook(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = build_synthetic_didi_finance_data()
    guidance = pd.DataFrame(
        [
            {
                "配置项": "数据性质",
                "示例值": "合成数据，仅用于 Forecast Lab 演示，不代表滴滴实际数据",
            },
            {"配置项": "Sheet", "示例值": "财务月度"},
            {"配置项": "时间字段", "示例值": "期间"},
            {"配置项": "目标字段", "示例值": "实际金额（元）"},
            {"配置项": "序列维度", "示例值": "业务线、财务科目"},
            {"配置项": "已知未来协变量", "示例值": "计划订单量（万单）、工作日数"},
            {"配置项": "历史滞后协变量", "示例值": "燃油价格指数"},
            {"配置项": "静态属性", "示例值": "城市覆盖数"},
            {"配置项": "增长率示例", "示例值": "经营增长假设：每预测期 0.30%"},
            {"配置项": "事件影响示例", "示例值": "五一与国庆：影响月份 5、10，按业务配置影响率"},
            {
                "配置项": "协变量缺失示例",
                "示例值": "计划订单数/燃油指数无历史字段时，使用手工情景协变量输出未来假设值并配置目标影响系数",
            },
            {"配置项": "推荐实验参数", "示例值": "月度，预测 6 期，回测 3 窗口，快速验证"},
        ]
    )
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        data.to_excel(writer, sheet_name="财务月度", index=False)
        guidance.to_excel(writer, sheet_name="实验说明", index=False)
        for worksheet in writer.book.worksheets:
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for column_cells in worksheet.columns:
                max_length = max(len(str(cell.value or "")) for cell in column_cells)
                worksheet.column_dimensions[column_cells[0].column_letter].width = min(
                    max(max_length + 2, 12), 42
                )
    return output_path
