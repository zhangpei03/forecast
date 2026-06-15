from __future__ import annotations

import pandas as pd


def build_item_id(data: pd.DataFrame, item_columns: list[str]) -> pd.Series:
    if not item_columns:
        raise ValueError("item_columns must not be empty")
    values = data[item_columns].fillna("未填写").astype(str)
    return values.apply(lambda row: " / ".join(part.strip() for part in row), axis=1)


def guess_mapping_columns(data: pd.DataFrame) -> dict[str, str | list[str]]:
    columns = [str(column) for column in data.columns]
    lower_map = {column: column.lower() for column in columns}

    timestamp_candidates = ["期间", "日期", "月份", "month", "period", "date", "timestamp"]
    target_candidates = ["实际值", "金额", "收入", "成本", "target", "actual", "value"]
    dimension_candidates = ["组织", "部门", "科目", "产品", "entity", "account", "product"]

    timestamp_column = _first_matching(columns, lower_map, timestamp_candidates) or columns[0]
    target_column = _first_matching(columns, lower_map, target_candidates) or columns[-1]
    item_columns = [
        column
        for column in columns
        if column not in {timestamp_column, target_column}
        and any(candidate.lower() in lower_map[column] for candidate in dimension_candidates)
    ]
    if not item_columns:
        item_columns = [
            column for column in columns if column not in {timestamp_column, target_column}
        ][:1]

    return {
        "timestamp_column": timestamp_column,
        "target_column": target_column,
        "item_columns": item_columns,
    }


def _first_matching(
    columns: list[str],
    lower_map: dict[str, str],
    candidates: list[str],
) -> str | None:
    for candidate in candidates:
        candidate_lower = candidate.lower()
        for column in columns:
            if candidate_lower in lower_map[column]:
                return column
    return None
