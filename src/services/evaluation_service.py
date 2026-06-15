from __future__ import annotations

import math

import pandas as pd

from src.domain.models import PredictionMetrics


def calculate_prediction_metrics(predictions: pd.DataFrame) -> PredictionMetrics:
    if predictions.empty:
        return PredictionMetrics(None, None, None, 0.0, None, None)

    actual = predictions["actual"].astype(float)
    forecast = predictions["forecast_p50"].astype(float)
    valid = actual.notna() & forecast.notna()
    if not valid.any():
        return PredictionMetrics(None, None, None, 0.0, None, None)

    actual = actual[valid]
    forecast = forecast[valid]
    error = forecast - actual
    absolute_error = error.abs()
    denominator = actual.abs().sum()
    wape = float(absolute_error.sum() / denominator) if denominator != 0 else None
    mae = float(absolute_error.mean())
    rmse = float(math.sqrt(((error) ** 2).mean()))
    bias_amount = float(error.sum())
    bias_rate = float(bias_amount / denominator) if denominator != 0 else None

    coverage = None
    if {"forecast_p10", "forecast_p90"}.issubset(predictions.columns):
        interval = predictions.loc[valid, ["forecast_p10", "forecast_p90"]].astype(float)
        covered = (interval["forecast_p10"] <= actual) & (actual <= interval["forecast_p90"])
        coverage = float(covered.mean()) if len(covered) else None

    return PredictionMetrics(wape, mae, rmse, bias_amount, bias_rate, coverage)


def evaluate_models(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in predictions.groupby("model", sort=False):
        metrics = calculate_prediction_metrics(group)
        model_type = group.get("model_type", pd.Series(["AutoGluon"])).iloc[0]
        rows.append(
            {
                "model": model,
                "model_type": model_type,
                "wape": metrics.wape,
                "mae": metrics.mae,
                "rmse": metrics.rmse,
                "bias_amount": metrics.bias_amount,
                "bias_rate": metrics.bias_rate,
                "coverage": metrics.coverage,
                "training_seconds": 0.0,
                "prediction_seconds": 0.0,
                "status": "完成",
            }
        )
    leaderboard = pd.DataFrame(rows)
    if leaderboard.empty:
        return leaderboard
    return (
        leaderboard.sort_values(
            by=["wape", "bias_rate_abs", "mae", "training_seconds"],
            key=lambda column: column.abs() if column.name == "bias_rate_abs" else column,
        )
        if False
        else _sort_leaderboard(leaderboard)
    )


def _sort_leaderboard(leaderboard: pd.DataFrame) -> pd.DataFrame:
    ranked = leaderboard.copy()
    ranked["bias_rate_abs"] = ranked["bias_rate"].abs()
    ranked = ranked.sort_values(
        ["wape", "bias_rate_abs", "mae", "training_seconds"],
        na_position="last",
    ).drop(columns=["bias_rate_abs"])
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked.reset_index(drop=True)


def evaluate_by_series(predictions: pd.DataFrame, model: str) -> pd.DataFrame:
    rows = []
    selected = predictions[predictions["model"] == model]
    for item_id, group in selected.groupby("item_id", sort=True):
        metrics = calculate_prediction_metrics(group)
        rows.append(
            {
                "item_id": item_id,
                "wape": metrics.wape,
                "mae": metrics.mae,
                "bias_rate": metrics.bias_rate,
                "coverage": metrics.coverage,
                "points": int(group.shape[0]),
            }
        )
    return pd.DataFrame(rows).sort_values("wape", ascending=False, na_position="last")


def evaluate_by_window(
    predictions: pd.DataFrame,
    *,
    best_model: str,
    best_baseline: str,
) -> pd.DataFrame:
    rows = []
    for window_id, group in predictions.groupby("window_id", sort=True):
        model_metrics = calculate_prediction_metrics(group[group["model"] == best_model])
        baseline_metrics = calculate_prediction_metrics(group[group["model"] == best_baseline])
        improved = (
            model_metrics.wape is not None
            and baseline_metrics.wape is not None
            and model_metrics.wape < baseline_metrics.wape
        )
        rows.append(
            {
                "window_id": window_id,
                "model_wape": model_metrics.wape,
                "baseline_wape": baseline_metrics.wape,
                "improved": improved,
            }
        )
    return pd.DataFrame(rows)


def choose_best_model(leaderboard: pd.DataFrame) -> pd.Series:
    if leaderboard.empty:
        raise ValueError("leaderboard must not be empty")
    ranked = leaderboard.copy()
    ranked["bias_rate_abs"] = ranked["bias_rate"].abs()
    ranked = ranked.sort_values(
        ["wape", "bias_rate_abs", "mae", "training_seconds"],
        na_position="last",
    )
    return ranked.iloc[0].drop(labels=["bias_rate_abs"])


def choose_best_baseline(leaderboard: pd.DataFrame) -> pd.Series | None:
    baseline = leaderboard[leaderboard["model_type"].eq("基线")]
    if baseline.empty:
        return None
    return choose_best_model(baseline)


def classify_stability(improved_windows: int, total_windows: int) -> str:
    if total_windows <= 1:
        return "不可判断"
    win_rate = improved_windows / total_windows
    if win_rate > 0.70:
        return "稳定领先"
    if win_rate >= 0.50:
        return "有一定优势"
    return "不稳定"


def calculate_improvement(best_wape: float | None, baseline_wape: float | None) -> float | None:
    if best_wape is None or baseline_wape in (None, 0):
        return None
    return float((baseline_wape - best_wape) / baseline_wape)


def build_business_conclusion(
    *,
    best_model: str,
    best_wape: float | None,
    best_baseline: str,
    baseline_wape: float | None,
    improved_windows: int,
    total_windows: int,
    high_risk_series_count: int,
) -> str:
    improvement = calculate_improvement(best_wape, baseline_wape)
    stability = classify_stability(improved_windows, total_windows)
    best_wape_text = _format_percent(best_wape)
    baseline_wape_text = _format_percent(baseline_wape)
    improvement_text = _format_percent(improvement)
    return (
        f"{best_model} 为本次最佳模型，聚合 WAPE 为 {best_wape_text}，"
        f"较最佳业务基线 {best_baseline} 的 {baseline_wape_text} 改善 {improvement_text}。"
        f"该模型在 {total_windows} 个回测窗口中有 {improved_windows} 个优于基线，"
        f"当前结果属于“{stability}”。"
        f"仍有 {high_risk_series_count} 条序列 WAPE 高于 30%，建议进一步检查异常值或补充业务驱动变量。"
    )


def _format_percent(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"
