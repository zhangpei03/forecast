from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

BRAND = "#4F46E5"
TEXT = "#101828"
MUTED = "#98A2B3"
DANGER = "#F04438"
BLUE = "#2E90FA"
GRID = "#EAECF0"


def apply_finance_layout(fig: go.Figure, *, height: int = 420) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 24, "r": 20, "t": 26, "b": 24},
        font={"family": "-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif", "size": 12},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.08, "x": 0, "xanchor": "left", "yanchor": "bottom"},
        modebar={"remove": ["lasso2d", "select2d", "autoScale2d"]},
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, tickfont={"color": "#667085"})
    fig.update_yaxes(gridcolor=GRID, zerolinecolor="#D0D5DD", tickfont={"color": "#667085"})
    return fig


def build_trend_figure(
    *,
    actuals: pd.DataFrame,
    backtest_predictions: pd.DataFrame,
    future_forecast: pd.DataFrame,
    model: str,
    item_id: str | None = None,
) -> go.Figure:
    actual_data = _filter_item(actuals, item_id)
    backtest = _filter_item(backtest_predictions[backtest_predictions["model"].eq(model)], item_id)
    future = _filter_item(future_forecast[future_forecast["model"].eq(model)], item_id)

    fig = go.Figure()
    if not future.empty:
        interval = future.sort_values("timestamp")
        fig.add_trace(
            go.Scatter(
                x=list(interval["timestamp"]) + list(interval["timestamp"])[::-1],
                y=list(interval["forecast_p90"]) + list(interval["forecast_p10"])[::-1],
                fill="toself",
                fillcolor="rgba(79,70,229,0.12)",
                line={"color": "rgba(255,255,255,0)"},
                hoverinfo="skip",
                name="P10-P90",
            )
        )
    if not actual_data.empty:
        fig.add_trace(
            go.Scatter(
                x=actual_data["timestamp"],
                y=actual_data["target"],
                mode="lines+markers",
                name="实际值",
                line={"color": TEXT, "width": 2.4},
                marker={"size": 5},
            )
        )
    if not backtest.empty:
        fig.add_trace(
            go.Scatter(
                x=backtest["timestamp"],
                y=backtest["forecast_p50"],
                mode="lines+markers",
                name="回测预测",
                line={"color": BRAND, "width": 2.5},
                marker={"size": 5},
                customdata=backtest[
                    ["actual", "error", "error_rate", "forecast_p10", "forecast_p90"]
                ],
                hovertemplate=(
                    "期间=%{x}<br>预测=%{y:,.0f}<br>实际=%{customdata[0]:,.0f}"
                    "<br>偏差=%{customdata[1]:,.0f}<br>偏差率=%{customdata[2]:.1%}"
                    "<br>P10=%{customdata[3]:,.0f}<br>P90=%{customdata[4]:,.0f}<extra></extra>"
                ),
            )
        )
    if not future.empty:
        fig.add_trace(
            go.Scatter(
                x=future["timestamp"],
                y=future["forecast_p50"],
                mode="lines+markers",
                name="未来预测",
                line={"color": BRAND, "width": 2.5, "dash": "dot"},
                marker={"size": 5},
            )
        )
        first_future = pd.to_datetime(future["timestamp"]).min()
        fig.add_vline(x=first_future, line_dash="dash", line_color="#D0D5DD")
    return apply_finance_layout(fig, height=420)


def build_deviation_bar_figure(
    predictions: pd.DataFrame, model: str, item_id: str | None = None
) -> go.Figure:
    data = _filter_item(predictions[predictions["model"].eq(model)], item_id).sort_values(
        "timestamp"
    )
    colors = [DANGER if value >= 0 else BLUE for value in data["error"]]
    fig = go.Figure(
        go.Bar(
            x=data["timestamp"],
            y=data["error"],
            marker_color=colors,
            name="偏差金额",
            customdata=data[["actual", "forecast_p50", "error_rate"]],
            hovertemplate=(
                "期间=%{x}<br>实际=%{customdata[0]:,.0f}<br>预测=%{customdata[1]:,.0f}"
                "<br>偏差=%{y:,.0f}<br>偏差率=%{customdata[2]:.1%}<extra></extra>"
            ),
        )
    )
    fig.add_hline(y=0, line_color=MUTED, line_width=1.2)
    return apply_finance_layout(fig, height=340)


def build_actual_vs_forecast_figure(
    predictions: pd.DataFrame,
    model: str,
    item_id: str | None = None,
) -> go.Figure:
    data = _filter_item(predictions[predictions["model"].eq(model)], item_id)
    fig = go.Figure()
    if data.empty:
        return apply_finance_layout(fig, height=340)
    min_value = float(min(data["actual"].min(), data["forecast_p50"].min()))
    max_value = float(max(data["actual"].max(), data["forecast_p50"].max()))
    fig.add_trace(
        go.Scatter(
            x=[min_value, max_value],
            y=[min_value, max_value],
            mode="lines",
            line={"color": MUTED, "dash": "dash"},
            name="完美预测",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data["actual"],
            y=data["forecast_p50"],
            mode="markers",
            marker={
                "color": BRAND,
                "size": 8,
                "opacity": 0.78,
                "line": {"color": "white", "width": 1},
            },
            text=data["item_id"].astype(str) + " · " + data["timestamp"].astype(str),
            name="预测点",
            hovertemplate="%{text}<br>实际=%{x:,.0f}<br>预测=%{y:,.0f}<extra></extra>",
        )
    )
    fig.update_xaxes(title="实际值")
    fig.update_yaxes(title="预测值")
    return apply_finance_layout(fig, height=340)


def _filter_item(data: pd.DataFrame, item_id: str | None) -> pd.DataFrame:
    if item_id and item_id != "全部序列":
        return data[data["item_id"].eq(item_id)].copy()
    if "target" in data.columns:
        return data.groupby("timestamp", as_index=False)["target"].sum()
    if data.empty:
        return data.copy()
    numeric_columns = [
        "actual",
        "forecast_mean",
        "forecast_p10",
        "forecast_p50",
        "forecast_p90",
        "error",
    ]
    agg_columns = {column: "sum" for column in numeric_columns if column in data.columns}
    aggregated = data.groupby(["timestamp", "model"], as_index=False).agg(agg_columns)
    if {"actual", "error"}.issubset(aggregated.columns):
        aggregated["error_rate"] = aggregated.apply(
            lambda row: row["error"] / abs(row["actual"]) if row["actual"] != 0 else pd.NA,
            axis=1,
        )
    aggregated["item_id"] = "全部序列"
    return aggregated
