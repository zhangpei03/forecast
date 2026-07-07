from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.exceptions import ValidationError
from src.services.mapping_service import build_item_id

YEAR_MONTH_CN_PATTERN = re.compile(r"^(\d{4})年第(\d{1,2})月$")
YYYYMM_PATTERN = re.compile(r"^\d{6}$")
NEGATIVE_PARENTHESES_PATTERN = re.compile(r"^\((.+)\)$")
NUMERIC_PATTERN = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def list_excel_sheets(path: Path) -> list[str]:
    workbook = pd.ExcelFile(path)
    return list(workbook.sheet_names)


def read_excel_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    data = pd.read_excel(path, sheet_name=sheet_name)
    if data.empty:
        raise ValidationError("当前 Sheet 没有可用记录", detail="SHEET_EMPTY")
    return data


def parse_timestamp(value: Any) -> pd.Timestamp:
    if pd.isna(value):
        return pd.NaT
    if isinstance(value, pd.Timestamp):
        return value.normalize()
    if isinstance(value, datetime):
        return pd.Timestamp(value).normalize()

    raw_value = str(value).strip()
    if not raw_value:
        return pd.NaT

    cn_match = YEAR_MONTH_CN_PATTERN.match(raw_value)
    if cn_match:
        year, month = cn_match.groups()
        return pd.Timestamp(year=int(year), month=int(month), day=1)

    if YYYYMM_PATTERN.match(raw_value):
        return pd.Timestamp(year=int(raw_value[:4]), month=int(raw_value[4:]), day=1)

    for date_format in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m"):
        try:
            parsed = datetime.strptime(raw_value, date_format)
            return pd.Timestamp(parsed).normalize()
        except ValueError:
            continue

    parsed = pd.to_datetime(raw_value, errors="coerce")
    return parsed.normalize() if not pd.isna(parsed) else pd.NaT


def parse_amount(value: Any) -> float:
    if pd.isna(value):
        return float("nan")
    if isinstance(value, int | float):
        return float(value)

    raw_value = str(value).strip()
    if raw_value == "":
        return float("nan")
    negative = False
    parentheses_match = NEGATIVE_PARENTHESES_PATTERN.match(raw_value)
    if parentheses_match:
        negative = True
        raw_value = parentheses_match.group(1)

    normalized = raw_value.replace(",", "")
    if not NUMERIC_PATTERN.fullmatch(normalized):
        return float("nan")
    try:
        parsed = float(normalized)
    except ValueError:
        return float("nan")
    return -parsed if negative else parsed


def normalize_finance_dataframe(
    data: pd.DataFrame,
    *,
    timestamp_column: str,
    target_column: str,
    item_columns: list[str],
    known_covariates: list[str] | None = None,
    past_covariates: list[str] | None = None,
    static_features: list[str] | None = None,
    duplicate_strategy: str = "sum",
    missing_strategy: str = "block",
) -> pd.DataFrame:
    if not item_columns:
        raise ValidationError("至少需要选择一个序列维度字段")
    required_columns = [timestamp_column, target_column, *item_columns]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValidationError(f"缺少必要字段：{', '.join(missing_columns)}")

    normalized = data.copy()
    normalized["timestamp"] = normalized[timestamp_column].map(parse_timestamp)
    normalized["target"] = normalized[target_column].map(parse_amount)
    for column in dict.fromkeys([*(known_covariates or []), *(past_covariates or [])]):
        if column in normalized:
            normalized[column] = normalized[column].map(parse_amount)
    normalized["item_id"] = build_item_id(normalized, item_columns)

    selected_columns = [
        "item_id",
        "timestamp",
        "target",
        *item_columns,
        *(known_covariates or []),
        *(past_covariates or []),
        *(static_features or []),
    ]
    existing_columns = list(
        dict.fromkeys(column for column in selected_columns if column in normalized)
    )
    normalized = normalized.loc[:, existing_columns]

    if duplicate_strategy != "block":
        normalized = aggregate_duplicates(normalized, duplicate_strategy)
    if missing_strategy != "block":
        normalized = fill_missing_targets(normalized, missing_strategy)
        normalized = fill_missing_covariates(
            normalized,
            list(dict.fromkeys([*(known_covariates or []), *(past_covariates or [])])),
            missing_strategy,
        )

    return normalized.sort_values(["item_id", "timestamp"]).reset_index(drop=True)


def aggregate_duplicates(data: pd.DataFrame, strategy: str) -> pd.DataFrame:
    group_columns = ["item_id", "timestamp"]
    value_columns = [column for column in data.columns if column not in group_columns]
    if strategy == "sum":
        return data.groupby(group_columns, as_index=False).agg(
            {column: "first" for column in value_columns if column != "target"}
            | {"target": lambda values: values.sum(min_count=1)}
        )
    if strategy == "last":
        return data.drop_duplicates(group_columns, keep="last")
    if strategy == "mean":
        return data.groupby(group_columns, as_index=False).agg(
            {column: "first" for column in value_columns if column != "target"} | {"target": "mean"}
        )
    raise ValidationError(f"未知重复记录处理方式：{strategy}")


def fill_missing_targets(data: pd.DataFrame, strategy: str) -> pd.DataFrame:
    if strategy == "zero":
        # 仅对历史行（target 有值 或 timestamp <= 历史最大日期）填充，未来预填行保留 NaN
        filled = data.copy()
        max_hist_ts = (
            data.loc[data["target"].notna()].groupby("item_id")["timestamp"].max().rename("_mhts")
        )
        merged = filled.join(max_hist_ts, on="item_id")
        is_future = filled["target"].isna() & (
            pd.to_datetime(merged["timestamp"]) > pd.to_datetime(merged["_mhts"])
        )
        filled.loc[~is_future, "target"] = filled.loc[~is_future, "target"].fillna(0)
        return filled.drop(columns=[], errors="ignore")
    if strategy == "ffill":
        filled = data.sort_values(["item_id", "timestamp"]).copy()
        max_hist_ts = (
            filled.loc[filled["target"].notna()]
            .groupby("item_id")["timestamp"]
            .max()
            .rename("_mhts")
        )
        merged_ts = filled.join(max_hist_ts, on="item_id")
        is_future = filled["target"].isna() & (
            pd.to_datetime(merged_ts["timestamp"]) > pd.to_datetime(merged_ts["_mhts"])
        )
        orig_future_target = filled.loc[is_future, "target"].copy()
        filled["target"] = filled.groupby("item_id")["target"].ffill()
        filled.loc[is_future, "target"] = orig_future_target
        return filled
    if strategy == "interpolate":
        filled = data.sort_values(["item_id", "timestamp"]).copy()
        max_hist_ts = (
            filled.loc[filled["target"].notna()]
            .groupby("item_id")["timestamp"]
            .max()
            .rename("_mhts")
        )
        merged_ts = filled.join(max_hist_ts, on="item_id")
        is_future = filled["target"].isna() & (
            pd.to_datetime(merged_ts["timestamp"]) > pd.to_datetime(merged_ts["_mhts"])
        )
        orig_future_target = filled.loc[is_future, "target"].copy()
        filled["target"] = filled.groupby("item_id")["target"].transform(
            lambda series: series.interpolate(method="linear")
        )
        filled.loc[is_future, "target"] = orig_future_target
        return filled
    raise ValidationError(f"未知缺失值处理方式：{strategy}")


def fill_missing_covariates(
    data: pd.DataFrame,
    columns: list[str],
    strategy: str,
) -> pd.DataFrame:
    """Fill historical covariate gaps without consuming future blank rows."""
    available = [column for column in columns if column in data.columns]
    if not available:
        return data

    filled = data.sort_values(["item_id", "timestamp"]).copy()
    historical_mask = filled["target"].notna()
    historical = filled.loc[historical_mask, ["item_id", *available]].copy()
    for column in available:
        values = pd.to_numeric(historical[column], errors="coerce")
        if strategy == "zero":
            historical[column] = values.fillna(0.0)
        elif strategy == "ffill":
            historical[column] = values.groupby(historical["item_id"]).transform(
                lambda series: series.ffill().bfill()
            )
        elif strategy == "interpolate":
            historical[column] = values.groupby(historical["item_id"]).transform(
                lambda series: series.interpolate(method="linear", limit_direction="both")
            )
        else:
            raise ValidationError(f"未知缺失值处理方式：{strategy}")
    filled.loc[historical_mask, available] = historical[available]
    return filled
