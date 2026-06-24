from __future__ import annotations

from pathlib import Path

from src.services.didi_example_service import (
    write_daily_example_workbook,
    write_example_workbook,
)


def main() -> None:
    monthly_path = write_example_workbook(Path("sample_data/didi_finance_forecast_example.xlsx"))
    print(f"Wrote synthetic Didi monthly finance example to {monthly_path}")
    daily_path = write_daily_example_workbook(
        Path("sample_data/didi_finance_forecast_daily_example.xlsx")
    )
    print(f"Wrote synthetic Didi daily finance example to {daily_path}")


if __name__ == "__main__":
    main()
