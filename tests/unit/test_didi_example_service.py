from pathlib import Path

from src.services.didi_example_service import (
    build_synthetic_didi_finance_data,
    write_example_workbook,
)


def test_build_synthetic_didi_finance_data_has_complete_monthly_driver_fields() -> None:
    data = build_synthetic_didi_finance_data()

    assert data.shape[0] == 384
    assert data["业务线"].nunique() == 4
    assert data["财务科目"].nunique() == 2
    assert data["期间"].min() == "2022-01"
    assert data["期间"].max() == "2025-12"
    assert (
        data[["实际金额（元）", "计划订单量（万单）", "工作日数", "燃油价格指数"]]
        .notna()
        .all()
        .all()
    )


def test_write_example_workbook_includes_data_and_guidance_sheets(tmp_path: Path) -> None:
    output_path = tmp_path / "didi_example.xlsx"

    write_example_workbook(output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
