# analyzer.py

from shariah_rules import check_sector, check_financial_ratios
from ml_predictor import ai_predict


def analyze(stock):
    """
    stock ต้องมี:
    - symbol
    - sector
    - interest_income
    - total_income
    - interest_debt
    - total_assets
    """

    # =========================
    # คำนวณ ratios
    # =========================
    interest_ratio = stock["interest_income"] / stock["total_income"]
    debt_ratio = stock["interest_debt"] / stock["total_assets"]
    non_halal_ratio = 0.0  # ถ้ายังไม่มี data

    # =========================
    # Rule-based (Shariah)
    # =========================
    sector_ok, sector_passed, sector_failed = check_sector(stock["sector"])

    fin_ok, fin_passed, fin_failed = check_financial_ratios(
        interest_ratio, debt_ratio, non_halal_ratio
    )

    passed_reasons = sector_passed + fin_passed
    failed_reasons = sector_failed + fin_failed

    # =========================
    # ตัดสินจากกฎก่อน (สำคัญสุด)
    # =========================
    if not sector_ok or not fin_ok:
        return {
            "symbol": stock["symbol"],
            "status": "HARAM",
            "confidence": 1.0,
            "failed_reasons": failed_reasons,
            "passed_reasons": passed_reasons
        }

    # =========================
    # ใช้ AI เพิ่ม (ถ้าไม่ตกกฎ)
    # =========================
    ai_status, confidence = ai_predict(stock)

    return {
        "symbol": stock["symbol"],
        "status": ai_status.upper(),
        "confidence": confidence,
        "failed_reasons": failed_reasons,
        "passed_reasons": passed_reasons
    }
