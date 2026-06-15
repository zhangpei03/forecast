import pandas as pd

from src.services.excel_service import parse_amount, parse_timestamp
from src.services.mapping_service import build_item_id


def test_parse_timestamp_accepts_supported_finance_formats() -> None:
    parsed = [
        parse_timestamp("2024-01-31"),
        parse_timestamp("2024/02/29"),
        parse_timestamp("2024-03"),
        parse_timestamp("202404"),
        parse_timestamp("2024年第5月"),
    ]

    assert [value.strftime("%Y-%m") for value in parsed] == [
        "2024-01",
        "2024-02",
        "2024-03",
        "2024-04",
        "2024-05",
    ]


def test_parse_amount_accepts_commas_and_parentheses_negative() -> None:
    assert parse_amount("1,234.50") == 1234.5
    assert parse_amount("(1,200)") == -1200
    assert pd.isna(parse_amount(""))


def test_parse_amount_rejects_unconfigured_currency_text() -> None:
    assert pd.isna(parse_amount("RMB 1,200"))


def test_build_item_id_joins_dimension_columns() -> None:
    frame = pd.DataFrame({"组织": ["总部"], "科目": ["人力成本"], "产品": ["全部"]})

    item_id = build_item_id(frame, ["组织", "科目", "产品"])

    assert item_id.iloc[0] == "总部 / 人力成本 / 全部"
