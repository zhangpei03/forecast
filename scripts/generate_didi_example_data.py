from __future__ import annotations

from pathlib import Path

from src.services.didi_example_service import write_example_workbook


def main() -> None:
    output_path = write_example_workbook(Path("sample_data/didi_finance_forecast_example.xlsx"))
    print(f"Wrote synthetic Didi finance example to {output_path}")


if __name__ == "__main__":
    main()
