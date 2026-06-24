from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.core.config import get_settings
from src.repositories.experiment_repository import ExperimentRepository
from src.services.chart_service import (
    build_actual_vs_forecast_figure,
    build_deviation_bar_figure,
    build_trend_figure,
)
from src.services.forecast_driver_service import (
    AVAILABILITY_KNOWN_FUTURE,
    CONFIG_TYPE_CALENDAR_FACTOR,
    CONFIG_TYPE_COVARIATE,
    CONFIG_TYPE_GROWTH_RATE,
    CONFIG_TYPE_SCENARIO_COVARIATE,
    FUTURE_VALUE_MEAN,
    normalize_driver_configs,
)
from src.storage.file_store import get_experiment_dir, read_json
from src.storage.parquet import read_parquet
from src.ui.components import page_header

settings = get_settings()
repository = ExperimentRepository(settings.database_path)


def _percent(value) -> str:
    return "—" if value is None or pd.isna(value) else f"{float(value) * 100:.1f}%"


def _finance_detail(data: pd.DataFrame) -> pd.DataFrame:
    columns = {
        "timestamp": "期间",
        "item_id": "item_id",
        "actual": "实际值",
        "forecast_p50": "P50 预测",
        "forecast_p10": "P10",
        "forecast_p90": "P90",
        "error": "偏差金额",
        "error_rate": "偏差率",
        "window_id": "回测窗口",
    }
    visible_columns = [column for column in columns if column in data.columns]
    return data.loc[:, visible_columns].rename(columns=columns)


def _kv_card(title: str, values: dict) -> None:
    rows = "".join(
        f'<div class="fl-kv"><span>{key}</span><strong>{value or "—"}</strong></div>'
        for key, value in values.items()
    )
    st.markdown(
        f'<div class="fl-card"><h3 class="fl-section-title">{title}</h3>{rows}</div>',
        unsafe_allow_html=True,
    )


experiment_id = st.query_params.get("experiment_id")
if not experiment_id:
    succeeded = [item for item in repository.list_experiments() if item.status.value == "SUCCEEDED"]
    experiment_id = succeeded[0].id if succeeded else None

if not experiment_id:
    page_header("结果分析", "暂无已完成实验。")
    st.info("完成一次评测后将在这里展示趋势、偏差和模型榜单。")
    st.stop()

summary = repository.get_experiment(experiment_id)
if summary is None:
    st.error("实验不存在。")
    st.stop()
if summary.status.value != "SUCCEEDED":
    st.warning("实验尚未完成，请先查看运行状态。")
    if st.button("查看运行状态"):
        st.query_params["experiment_id"] = experiment_id
        st.switch_page("pages/run_status.py")
    st.stop()

experiment_dir = get_experiment_dir(settings, experiment_id)
results_dir = experiment_dir / "results"
normalized = read_parquet(experiment_dir / "normalized_data.parquet")
leaderboard = read_parquet(results_dir / "leaderboard.parquet")
backtest = read_parquet(results_dir / "backtest_predictions.parquet")
future = read_parquet(results_dir / "future_forecast.parquet")
future_driver_assumptions = (
    read_parquet(results_dir / "future_driver_assumptions.parquet")
    if (results_dir / "future_driver_assumptions.parquet").exists()
    else pd.DataFrame()
)
series_metrics = read_parquet(results_dir / "series_metrics.parquet")
window_metrics = read_parquet(results_dir / "window_metrics.parquet")
conclusion = read_json(results_dir / "conclusion.json").get("conclusion", "暂无结论")

page_header(
    summary.name,
    f"数据范围：{summary.data_start} 至 {summary.data_end} · {summary.item_count} 条序列 · 预测未来 {summary.config.get('prediction_length')} 期",
    badge="评测完成",
)

col_a, col_b = st.columns([0.75, 0.25])
with col_b:
    export_info = read_json(results_dir / "export.json")
    export_path = Path(export_info.get("path", ""))
    if export_path.exists():
        st.download_button(
            "导出评测报告",
            data=export_path.read_bytes(),
            file_name=export_path.name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            use_container_width=True,
        )

st.markdown(
    f"""
    <div class="fl-conclusion">
      <div class="fl-icon">✓</div>
      <div>
        <div class="fl-eyebrow">评测结论</div>
        <div class="fl-title">{summary.best_model or "最佳模型"} 与业务基线完成统一回测比较</div>
        <div class="fl-desc">{conclusion}</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_model = summary.best_model or (leaderboard["model"].iloc[0] if not leaderboard.empty else "")
best_baseline = leaderboard[leaderboard["model_type"].eq("基线")].sort_values("wape").head(1)
baseline_name = best_baseline["model"].iloc[0] if not best_baseline.empty else "—"
model_rows = leaderboard[leaderboard["model"].eq(metric_model)]
metric_row = model_rows.iloc[0] if not model_rows.empty else None

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("最佳模型 WAPE", _percent(summary.best_wape), delta="优于最佳基线")
c2.metric("最佳基线 WAPE", _percent(summary.baseline_wape), delta=baseline_name)
c3.metric("相对基线提升", _percent(summary.improvement_rate), delta="目标 ≥10%")
c4.metric("Bias Rate", _percent(metric_row["bias_rate"] if metric_row is not None else None))
c5.metric("P10-P90 覆盖率", _percent(metric_row["coverage"] if metric_row is not None else None))

tabs = st.tabs(["评测总览", "趋势分析", "偏差分析", "模型榜单", "数据与配置"])

with tabs[0]:
    c_main, c_side = st.columns([0.66, 0.34])
    with c_main:
        st.markdown("### 实际值与预测趋势")
        st.plotly_chart(
            build_trend_figure(
                actuals=normalized,
                backtest_predictions=backtest,
                future_forecast=future,
                model=metric_model,
            ),
            use_container_width=True,
            config={"displaylogo": False},
            key="overview_trend_chart",
        )
    with c_side:
        st.markdown("### 风险与关注项")
        for _, row in series_metrics.head(3).iterrows():
            st.markdown(
                f"""
                <div class="fl-risk">
                  <strong>{row["item_id"]}</strong>
                  <p>序列 WAPE 为 {_percent(row["wape"])}，建议核查异常值或补充业务变量。</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    c_left, c_right = st.columns([0.58, 0.42])
    with c_left:
        st.markdown("### 回测窗口表现")
        st.dataframe(window_metrics, use_container_width=True, hide_index=True)
    with c_right:
        st.markdown("### 模型与基线 Top 5")
        st.dataframe(leaderboard.head(5), use_container_width=True, hide_index=True)

with tabs[1]:
    item_options = ["全部序列", *sorted(backtest["item_id"].dropna().astype(str).unique().tolist())]
    selected_item = st.selectbox("item_id", item_options)
    selected_model = st.selectbox("模型", leaderboard["model"].tolist(), key="trend_model")
    st.plotly_chart(
        build_trend_figure(
            actuals=normalized,
            backtest_predictions=backtest,
            future_forecast=future,
            model=selected_model,
            item_id=selected_item,
        ),
        use_container_width=True,
        config={"displaylogo": False},
        key="trend_detail_chart",
    )
    detail = backtest[backtest["model"].eq(selected_model)]
    if selected_item != "全部序列":
        detail = detail[detail["item_id"].eq(selected_item)]
    st.dataframe(_finance_detail(detail), use_container_width=True, hide_index=True)

with tabs[2]:
    selected_model = st.selectbox("偏差模型", leaderboard["model"].tolist(), key="deviation_model")
    deviation_items = [
        "全部序列",
        *sorted(backtest["item_id"].dropna().astype(str).unique().tolist()),
    ]
    selected_item = st.selectbox("偏差序列", deviation_items)
    m1, m2, m3, m4, m5 = st.columns(5)
    selected = backtest[backtest["model"].eq(selected_model)]
    if selected_item != "全部序列":
        selected = selected[selected["item_id"].eq(selected_item)]
    m1.metric("总偏差金额", f"{selected['error'].sum():,.0f}")
    m2.metric("Bias Rate", _percent(selected["error"].sum() / selected["actual"].abs().sum()))
    m3.metric("最大高估期间", _percent(selected["error_rate"].max()))
    m4.metric("最大低估期间", _percent(selected["error_rate"].min()))
    m5.metric("超过阈值期间", int((selected["error_rate"].abs() > 0.10).sum()))
    g1, g2 = st.columns(2)
    g1.plotly_chart(
        build_deviation_bar_figure(backtest, selected_model, selected_item),
        use_container_width=True,
        key="deviation_bar_chart",
    )
    g2.plotly_chart(
        build_actual_vs_forecast_figure(backtest, selected_model, selected_item),
        use_container_width=True,
        key="deviation_scatter_chart",
    )
    st.dataframe(_finance_detail(selected), use_container_width=True, hide_index=True)

with tabs[3]:
    st.markdown("### 模型评测排行榜")
    st.dataframe(leaderboard, use_container_width=True, hide_index=True)

with tabs[4]:
    c1, c2, c3 = st.columns(3)
    with c1:
        _kv_card(
            "原始数据",
            {
                "文件": Path(summary.source_file or "").name,
                "Sheet": summary.sheet_name,
                "记录数": summary.row_count,
            },
        )
    with c2:
        _kv_card(
            "字段映射",
            {
                "时间字段": summary.timestamp_column,
                "目标字段": summary.target_column,
                "序列维度": " + ".join(summary.item_columns),
                "频率": summary.config.get("freq"),
            },
        )
    with c3:
        _kv_card(
            "预测参数",
            {
                "预测周期": summary.config.get("prediction_length"),
                "回测窗口": summary.config.get("num_val_windows"),
                "训练模式": summary.config.get("preset"),
                "时间预算": summary.config.get("time_limit_seconds"),
            },
        )
    driver_configs = normalize_driver_configs(summary.config.get("driver_configs", []))
    if driver_configs:
        st.markdown("### 预测驱动配置")
        rows = []
        for driver in driver_configs:
            if driver.config_type == CONFIG_TYPE_COVARIATE:
                rows.append(
                    {
                        "名称": driver.name,
                        "类型": "协变量",
                        "字段": driver.column,
                        "可用性": "已知未来"
                        if driver.availability == AVAILABILITY_KNOWN_FUTURE
                        else "历史滞后",
                        "未来值规则": "历史均值延续"
                        if driver.future_value_strategy == FUTURE_VALUE_MEAN
                        else "历史末值延续",
                    }
                )
            elif driver.config_type == CONFIG_TYPE_GROWTH_RATE:
                rows.append(
                    {
                        "名称": driver.name,
                        "类型": "增长率",
                        "字段": "—",
                        "可用性": "全部候选模型",
                        "未来值规则": f"每预测期 {float(driver.growth_rate or 0) * 100:.2f}%",
                    }
                )
            elif driver.config_type == CONFIG_TYPE_CALENDAR_FACTOR:
                rows.append(
                    {
                        "名称": driver.name,
                        "类型": "节假日/事件影响",
                        "字段": "—",
                        "可用性": "影响月份："
                        + "、".join(f"{month}月" for month in driver.impact_months),
                        "未来值规则": f"影响率 {float(driver.impact_rate or 0) * 100:.2f}%",
                    }
                )
            elif driver.config_type == CONFIG_TYPE_SCENARIO_COVARIATE:
                rows.append(
                    {
                        "名称": driver.name,
                        "类型": "手工情景协变量",
                        "字段": "无历史字段",
                        "可用性": "输出未来假设",
                        "未来值规则": (
                            f"基准 {float(driver.base_value or 0):,.2f}，"
                            f"每期 {float(driver.scenario_growth_rate or 0) * 100:.2f}%，"
                            f"影响系数 {float(driver.effect_rate or 0) * 100:.2f}%"
                        ),
                    }
                )
        st.dataframe(rows, use_container_width=True, hide_index=True)
    if not future_driver_assumptions.empty:
        st.markdown("### 未来手工驱动假设")
        st.dataframe(future_driver_assumptions, use_container_width=True, hide_index=True)
    st.markdown("### 数据质量摘要")
    profile = read_json(experiment_dir / "data_profile.json")
    st.dataframe(pd.DataFrame(profile.get("issues", [])), use_container_width=True, hide_index=True)
    with st.expander("运行日志", expanded=False):
        log_path = experiment_dir / "run.log"
        st.code(log_path.read_text(encoding="utf-8")[-8000:] if log_path.exists() else "暂无日志")
