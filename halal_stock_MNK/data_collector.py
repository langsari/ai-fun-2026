# data_collector.py
import csv

def get_stock_data(symbol):
    with open("data/sample_stocks.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["symbol"] == symbol:
                return {
                    "symbol": symbol,
                    "sector": row["sector"],
                    "interest_income": float(row["interest_income"]),
                    "total_income": float(row["total_income"]),
                    "interest_debt": float(row["interest_debt"]),
                    "total_assets": float(row["total_assets"])
                }
    return None
