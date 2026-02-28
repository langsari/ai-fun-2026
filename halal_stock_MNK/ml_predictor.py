import joblib
import os
from shariah_rules import check_sector, check_financial_ratios

MODEL_PATH = os.path.join("model", "halal_model.pkl")
model = joblib.load(MODEL_PATH)

def ai_analyze(stock, sector):
    interest_ratio = stock["interest_income"] / stock["total_income"]
    debt_ratio = stock["interest_debt"] / stock["total_assets"]
    non_halal_ratio = 0.0

    # AI prediction
    X = [[interest_ratio, debt_ratio, non_halal_ratio]]
    prediction = model.predict(X)[0]
    confidence = max(model.predict_proba(X)[0])

    # Rule-based checks
    sector_ok, sector_reason = check_sector(sector)
    financial_ok, financial_reasons = check_financial_ratios(
        interest_ratio, debt_ratio, non_halal_ratio
    )

    reasons = [sector_reason] + financial_reasons

    if not sector_ok or not financial_ok:
        final_status = "HARAM"
    else:
        final_status = "HALAL"

    return final_status, confidence, reasons
