from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    output_dir = Path("sample_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    periods = pd.date_range("2023-01-01", periods=36, freq="MS")
    rows = []
    organizations = ["总部", "主站事业部", "商业化事业部", "海外事业部"]
    accounts = ["营业收入", "营业成本", "人力成本"]
    products = ["全部", "核心产品", "新业务"]
    for org_index, organization in enumerate(organizations):
        for account_index, account in enumerate(accounts):
            for product_index, product in enumerate(products):
                base = 8_000_000 + org_index * 900_000 + account_index * 500_000
                season = 1 + product_index * 0.04
                for period_index, period in enumerate(periods):
                    trend = period_index * (65_000 + org_index * 8_000)
                    seasonal = 260_000 * ((period.month % 12) / 12)
                    value = (base + trend + seasonal) * season
                    if account != "营业收入":
                        value *= 0.62 if account == "营业成本" else 0.28
                    rows.append(
                        {
                            "期间": period.strftime("%Y-%m"),
                            "组织": organization,
                            "科目": account,
                            "产品": product,
                            "实际值": round(value, 2),
                            "工作日数": 21 - int(period.month in {1, 2, 10}),
                            "预算人数": 120 + org_index * 18 + product_index * 5,
                            "是否春节": int(period.month == 2),
                        }
                    )

    demo = pd.DataFrame(rows)
    invalid = demo.head(30).copy()
    invalid["实际值"] = invalid["实际值"].astype(object)
    invalid.loc[0, "期间"] = "bad-date"
    invalid.loc[1, "实际值"] = "RMB 1200"
    invalid = pd.concat([invalid, invalid.iloc[[2]]], ignore_index=True)

    with pd.ExcelWriter(output_dir / "finance_monthly_demo.xlsx", engine="openpyxl") as writer:
        demo.to_excel(writer, sheet_name="月度实际", index=False)
    with pd.ExcelWriter(output_dir / "finance_monthly_invalid.xlsx", engine="openpyxl") as writer:
        invalid.to_excel(writer, sheet_name="月度实际", index=False)
    print(f"Wrote demo workbooks to {output_dir}")


if __name__ == "__main__":
    main()
