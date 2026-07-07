import pandas as pd

from src.services.excel_service import normalize_finance_dataframe, parse_amount, parse_timestamp
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


def test_parse_amount_accepts_scientific_notation_from_excel() -> None:
    assert parse_amount("3.203280158882696E-4") == 0.0003203280158882696
    assert parse_amount("-1.5e+3") == -1500.0


def test_build_item_id_joins_dimension_columns() -> None:
    frame = pd.DataFrame({"组织": ["总部"], "科目": ["人力成本"], "产品": ["全部"]})

    item_id = build_item_id(frame, ["组织", "科目", "产品"])

    assert item_id.iloc[0] == "总部 / 人力成本 / 全部"


def test_normalize_finance_dataframe_parses_selected_covariates_as_numbers() -> None:
    raw = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-02-01"],
            "account": ["收入", "收入"],
            "amount": ["1,000", "1,100"],
            "workdays": ["20", "19"],
        }
    )

    normalized = normalize_finance_dataframe(
        raw,
        timestamp_column="date",
        target_column="amount",
        item_columns=["account"],
        known_covariates=["workdays"],
    )

    assert normalized["workdays"].tolist() == [20.0, 19.0]


def test_sum_duplicate_strategy_preserves_future_blank_targets() -> None:
    raw = pd.DataFrame(
        {
            "date": ["2026-06-24", "2026-06-25"],
            "city": ["上海", "上海"],
            "gmv": [100.0, None],
            "rain": [0, 1],
        }
    )

    normalized = normalize_finance_dataframe(
        raw,
        timestamp_column="date",
        target_column="gmv",
        item_columns=["city"],
        known_covariates=["rain"],
        duplicate_strategy="sum",
    )

    assert normalized["target"].iloc[0] == 100.0
    assert pd.isna(normalized["target"].iloc[1])
    assert normalized["rain"].tolist() == [0.0, 1.0]


def test_interpolate_fills_historical_covariates_but_preserves_future_blanks() -> None:
    raw = pd.DataFrame(
        {
            "date": ["2026-06-22", "2026-06-23", "2026-06-24", "2026-06-25"],
            "city": ["上海"] * 4,
            "gmv": [100.0, 110.0, 120.0, None],
            "rate": [1.0, "-", 3.0, None],
        }
    )

    normalized = normalize_finance_dataframe(
        raw,
        timestamp_column="date",
        target_column="gmv",
        item_columns=["city"],
        known_covariates=["rate"],
        duplicate_strategy="sum",
        missing_strategy="interpolate",
    )

    assert normalized["rate"].iloc[:3].tolist() == [1.0, 2.0, 3.0]
    assert pd.isna(normalized["rate"].iloc[3])
