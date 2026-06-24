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


def build_synthetic_didi_daily_finance_data(
    *,
    start: str = "2023-06-01",
    days: int = 1095,
    seed: int = 20260624,
) -> pd.DataFrame:
    """构建确定性的滴滴出行天级合成财务数据，用于天级预测演示。

    数据覆盖近三年（默认 1095 天），按 业务线 × 财务科目 拆分序列，并显式引入
    天气、城市覆盖、节假日、行业竞争系数等影响因子。毛利率以"软约束 + 波动"方式
    体现：每条业务线设定目标毛利率，运营成本围绕"收入 ×（1 - 目标毛利率）"生成，
    再叠加各因子扰动，使实际毛利率在目标附近波动。
    """

    dates = pd.date_range(start, periods=days, freq="D")
    business_lines = {
        # take_rate 单位：元/单；target_gross_margin：目标毛利率（软约束）
        "网约车": {
            "orders": 3200.0,
            "take_rate": 10.8,
            "target_gross_margin": 0.30,
            "cities": 295,
            "weather_sensitivity": 0.22,
            "competition_sensitivity": 0.18,
            "growth": 0.00035,
        },
        "顺风车": {
            "orders": 1050.0,
            "take_rate": 8.9,
            "target_gross_margin": 0.39,
            "cities": 210,
            "weather_sensitivity": 0.30,
            "competition_sensitivity": 0.12,
            "growth": 0.00055,
        },
        "企业出行": {
            "orders": 540.0,
            "take_rate": 12.4,
            "target_gross_margin": 0.35,
            "cities": 78,
            "weather_sensitivity": 0.10,
            "competition_sensitivity": 0.08,
            "growth": 0.00040,
        },
        "两轮车": {
            "orders": 1480.0,
            "take_rate": 4.2,
            "target_gross_margin": 0.42,
            "cities": 175,
            "weather_sensitivity": 0.45,
            "competition_sensitivity": 0.15,
            "growth": 0.00025,
        },
    }
    rng = np.random.default_rng(seed)
    holiday_dates = _build_holiday_calendar(dates)
    spring_festival_dates = _build_spring_festival_dates(dates)
    rows: list[dict[str, object]] = []

    for business_index, (business_line, parameters) in enumerate(business_lines.items()):
        target_margin = float(parameters["target_gross_margin"])
        for day_index, day in enumerate(dates):
            is_holiday = int(day in holiday_dates)
            is_spring_festival = int(day in spring_festival_dates)
            is_workday = int(day.dayofweek < 5 and not is_holiday)

            # 天气：好天概率随季节变化（夏秋偏好，冬季雨雪偏多）。
            good_weather_prob = float(
                np.clip(0.74 + 0.12 * np.sin((day.dayofyear / 365.0) * 2 * np.pi), 0.45, 0.95)
            )
            is_good_weather = int(rng.random() < good_weather_prob)
            # 好天利于出行，雨雪天压制订单（两轮车受影响最大）。
            weather_factor = (
                1.0 + parameters["weather_sensitivity"] * 0.12
                if is_good_weather
                else 1.0 - parameters["weather_sensitivity"]
            )

            # 行业竞争系数：1.0 为基准，>1 表示竞争加剧（补贴战），压制 take_rate 与订单。
            competition_index = float(
                np.clip(
                    1.0
                    + 0.10 * np.sin(day_index / 90.0)
                    + 0.06 * np.sin(day_index / 30.0)
                    + rng.normal(0, 0.02),
                    0.80,
                    1.30,
                )
            )
            competition_factor = 1.0 - parameters["competition_sensitivity"] * (
                competition_index - 1.0
            )

            # 城市覆盖随季度扩张。
            city_coverage = int(parameters["cities"] + (day_index // 90) * (1 + business_index))

            # 周内与年度季节性。
            weekday_factor = 1.10 if day.dayofweek >= 5 else 0.97
            yearly_seasonal = 1 + 0.07 * np.sin((day.dayofyear / 365.0) * 2 * np.pi)
            holiday_factor = 0.78 if is_spring_festival else (1.12 if is_holiday else 1.0)

            plan_orders = (
                parameters["orders"]
                * (1 + day_index * parameters["growth"])
                * yearly_seasonal
                * weekday_factor
                * holiday_factor
                * (1 + rng.normal(0, 0.015))
            )
            actual_orders = (
                plan_orders * weather_factor * competition_factor * (1 + rng.normal(0, 0.025))
            )
            actual_orders = max(actual_orders, 1.0)

            # 收入：竞争加剧会拉低单均抽成。
            effective_take_rate = parameters["take_rate"] * (1.0 - 0.08 * (competition_index - 1.0))
            revenue = actual_orders * effective_take_rate * (1 + rng.normal(0, 0.012))

            # 成本：软约束——围绕"收入 ×（1 - 目标毛利率）"波动，并受天气/竞争扰动。
            cost_pressure = (
                1.0
                + 0.05 * (1 - weather_factor)
                + 0.04 * (competition_index - 1.0)
                + rng.normal(0, 0.015)
            )
            operating_cost = revenue * (1 - target_margin) * cost_pressure
            operating_cost = float(np.clip(operating_cost, revenue * 0.40, revenue * 0.96))

            shared_fields = {
                "期间": day.strftime("%Y-%m-%d"),
                "业务线": business_line,
                "计划订单量（万单）": round(float(plan_orders), 3),
                "好天气": is_good_weather,
                "好天气概率": round(good_weather_prob, 3),
                "竞争强度指数": round(competition_index, 3),
                "城市覆盖数": city_coverage,
                "是否节假日": is_holiday,
                "是否春节": is_spring_festival,
                "是否工作日": is_workday,
                "目标毛利率": target_margin,
            }
            for account, amount in (("订单收入", revenue), ("运营成本", operating_cost)):
                rows.append(
                    {
                        **shared_fields,
                        "财务科目": account,
                        "实际金额（元）": round(float(amount), 2),
                    }
                )

    columns = [
        "期间",
        "业务线",
        "财务科目",
        "实际金额（元）",
        "计划订单量（万单）",
        "好天气",
        "好天气概率",
        "竞争强度指数",
        "城市覆盖数",
        "是否节假日",
        "是否春节",
        "是否工作日",
        "目标毛利率",
    ]
    return pd.DataFrame(rows)[columns]


def _build_holiday_calendar(dates: pd.DatetimeIndex) -> set[pd.Timestamp]:
    """中国主要法定节假日（含调休近似）：元旦、春节、清明、五一、端午、国庆。"""

    holidays: set[pd.Timestamp] = set()
    years = sorted({int(day.year) for day in dates})
    fixed_ranges = {
        # (month, start_day, length)
        "元旦": (1, 1, 3),
        "清明": (4, 4, 3),
        "五一": (5, 1, 5),
        "端午": (6, 22, 3),
        "国庆": (10, 1, 7),
    }
    for year in years:
        for _, (month, start_day, length) in fixed_ranges.items():
            start = pd.Timestamp(year=year, month=month, day=start_day)
            holidays.update(pd.date_range(start, periods=length, freq="D"))
    holidays.update(_build_spring_festival_dates(dates))
    return {day for day in holidays if day in dates}


def _build_spring_festival_dates(dates: pd.DatetimeIndex) -> set[pd.Timestamp]:
    """春节假期近似（农历正月初一前后 7 天），按年份手工锚定。"""

    spring_festival_anchors = {
        2022: "2022-01-31",
        2023: "2023-01-21",
        2024: "2024-02-10",
        2025: "2025-01-29",
        2026: "2026-02-17",
        2027: "2027-02-06",
    }
    result: set[pd.Timestamp] = set()
    years = sorted({int(day.year) for day in dates})
    for year in years:
        anchor = spring_festival_anchors.get(year)
        if anchor is None:
            continue
        start = pd.Timestamp(anchor)
        result.update(pd.date_range(start, periods=7, freq="D"))
    return {day for day in result if day in dates}


def write_daily_example_workbook(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = build_synthetic_didi_daily_finance_data()
    guidance = pd.DataFrame(
        [
            {
                "配置项": "数据性质",
                "示例值": "合成数据，仅用于 Forecast Lab 演示，不代表滴滴实际数据",
            },
            {"配置项": "Sheet", "示例值": "财务日度"},
            {"配置项": "时间粒度", "示例值": "天级，覆盖近 3 年（约 1095 天）"},
            {"配置项": "时间字段", "示例值": "期间（YYYY-MM-DD）"},
            {"配置项": "目标字段", "示例值": "实际金额（元）"},
            {"配置项": "序列维度", "示例值": "业务线、财务科目"},
            {
                "配置项": "已知未来协变量",
                "示例值": "计划订单量（万单）、好天气概率、竞争强度指数、城市覆盖数、是否节假日、是否春节、是否工作日",
            },
            {
                "配置项": "毛利率约束",
                "示例值": "目标毛利率为软约束，运营成本围绕目标波动并受天气/竞争扰动",
            },
            {
                "配置项": "影响因子",
                "示例值": "天气（好天/雨雪）、城市覆盖、节假日、行业竞争系数、工作日",
            },
            {"配置项": "推荐实验参数", "示例值": "日度，预测 30 天，回测 3 窗口，标准评测"},
        ]
    )
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        data.to_excel(writer, sheet_name="财务日度", index=False)
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
