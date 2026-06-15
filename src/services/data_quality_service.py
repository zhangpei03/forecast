from __future__ import annotations

from collections import Counter

import pandas as pd

from src.core.constants import MIN_TRAIN_LENGTH, RECOMMENDED_TRAIN_LENGTH
from src.domain.models import DataProfile, QualityIssue


def detect_series_frequency(timestamps: pd.Series | pd.DatetimeIndex) -> str:
    series = pd.Series(pd.to_datetime(timestamps).dropna().sort_values().unique())
    if len(series) < 2:
        return "M"
    deltas = series.diff().dropna().dt.days
    median_days = float(deltas.median())
    if 27 <= median_days <= 32:
        return "M"
    if 6 <= median_days <= 8:
        return "W"
    return "D"


def detect_missing_periods(data: pd.DataFrame, freq: str) -> dict[str, list[str]]:
    gaps: dict[str, list[str]] = {}
    pandas_freq = _to_pandas_freq(freq)
    for item_id, group in data.dropna(subset=["timestamp"]).groupby("item_id"):
        timestamps = pd.to_datetime(group["timestamp"]).sort_values()
        if timestamps.empty:
            continue
        full_range = pd.date_range(timestamps.iloc[0], timestamps.iloc[-1], freq=pandas_freq)
        missing = sorted(set(full_range) - set(timestamps))
        if missing:
            gaps[str(item_id)] = [_format_period(value, freq) for value in missing]
    return gaps


def profile_normalized_data(
    data: pd.DataFrame,
    *,
    freq: str | None = None,
    prediction_length: int = 3,
) -> DataProfile:
    detected_freq = freq or detect_series_frequency(data["timestamp"])
    issues: list[QualityIssue] = []

    invalid_timestamps = int(data["timestamp"].isna().sum())
    if invalid_timestamps:
        issues.append(
            QualityIssue(
                "TIME_PARSE_FAILED",
                "blocking",
                "存在无法解析的时间字段记录，需修正后再训练。",
                invalid_timestamps,
            )
        )

    invalid_targets = int(data["target"].isna().sum())
    if invalid_targets:
        issues.append(
            QualityIssue(
                "TARGET_MISSING_OR_PARSE_FAILED",
                "blocking",
                "目标值存在缺失或非数值记录，需选择处理方式。",
                invalid_targets,
            )
        )

    duplicate_count = int(data.duplicated(["item_id", "timestamp"]).sum())
    if duplicate_count:
        issues.append(
            QualityIssue(
                "DUPLICATE_SERIES_TIME",
                "blocking",
                "同一序列在同一期间存在重复记录。",
                duplicate_count,
            )
        )

    series_lengths = data.dropna(subset=["timestamp"]).groupby("item_id")["timestamp"].nunique()
    too_short = series_lengths[series_lengths < MIN_TRAIN_LENGTH[detected_freq]]
    if not too_short.empty:
        issues.append(
            QualityIssue(
                "SERIES_TOO_SHORT",
                "blocking",
                "部分序列长度低于最低可训练要求。",
                int(too_short.shape[0]),
                [str(item_id) for item_id in too_short.head(5).index],
            )
        )

    short_but_trainable = series_lengths[
        (series_lengths >= MIN_TRAIN_LENGTH[detected_freq])
        & (series_lengths < RECOMMENDED_TRAIN_LENGTH[detected_freq])
    ]
    if not short_but_trainable.empty:
        issues.append(
            QualityIssue(
                "SERIES_HISTORY_SHORT",
                "warning",
                "部分序列历史长度偏短，模型排名可能不稳定。",
                int(short_but_trainable.shape[0]),
                [str(item_id) for item_id in short_but_trainable.head(5).index],
            )
        )

    zero_ratio = float((data["target"] == 0).mean()) if len(data) else 0
    if zero_ratio > 0.30:
        issues.append(
            QualityIssue(
                "ZERO_RATIO_HIGH",
                "warning",
                "目标值零值比例超过 30%，可能是稀疏序列。",
                int((data["target"] == 0).sum()),
            )
        )

    negative_count = int((data["target"] < 0).sum())
    if negative_count:
        issues.append(
            QualityIssue(
                "NEGATIVE_VALUES",
                "info",
                "目标值包含负数，主指标将使用 WAPE 而非 MAPE。",
                negative_count,
            )
        )

    gaps = detect_missing_periods(data, detected_freq)
    if gaps:
        issues.append(
            QualityIssue(
                "MISSING_PERIODS",
                "warning",
                "部分序列存在时间断档。",
                sum(len(values) for values in gaps.values()),
                [f"{item}: {', '.join(values[:3])}" for item, values in list(gaps.items())[:5]],
            )
        )

    blocking_count = sum(1 for issue in issues if issue.severity == "blocking")
    warning_count = sum(1 for issue in issues if issue.severity == "warning")
    clean_timestamps = pd.to_datetime(data["timestamp"]).dropna()

    return DataProfile(
        row_count=int(data.shape[0]),
        item_count=int(data["item_id"].nunique()) if "item_id" in data else 0,
        data_start=_format_period(clean_timestamps.min(), detected_freq)
        if not clean_timestamps.empty
        else None,
        data_end=_format_period(clean_timestamps.max(), detected_freq)
        if not clean_timestamps.empty
        else None,
        average_series_length=float(series_lengths.mean()) if not series_lengths.empty else 0,
        frequency=detected_freq,
        blocking_issue_count=blocking_count,
        warning_count=warning_count,
        issues=issues,
    )


def estimate_supported_backtest_windows(
    data: pd.DataFrame,
    *,
    prediction_length: int,
    requested_windows: int,
) -> int:
    shortest_length = int(data.groupby("item_id")["timestamp"].nunique().min())
    max_windows = max((shortest_length - prediction_length) // prediction_length, 1)
    return min(requested_windows, max_windows)


def summarize_issue_counts(issues: list[QualityIssue]) -> Counter[str]:
    return Counter(issue.severity for issue in issues)


def _to_pandas_freq(freq: str) -> str:
    return {"M": "MS", "W": "W-MON", "D": "D"}[freq]


def _format_period(value: pd.Timestamp, freq: str) -> str:
    if pd.isna(value):
        return ""
    if freq == "M":
        return pd.Timestamp(value).strftime("%Y-%m")
    if freq == "W":
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    return pd.Timestamp(value).strftime("%Y-%m-%d")
