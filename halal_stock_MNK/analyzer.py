# analyzer.py
from shariah_rules import check_shariah
from ml_predictor import ai_predict

def analyze(stock):
    status, reasons = check_shariah(stock)

    # ถ้าฮารอมชัด ไม่ต้องใช้ AI
    if status == "Haram":
        return {
            "symbol": stock["symbol"],
            "status": status,
            "reasons": reasons,
            "confidence": 1.0
        }

    # ถ้า Halal หรือ Doubtful → ให้ AI ประเมินเพิ่ม
    ai_status, confidence = ai_predict(stock)

    return {
        "symbol": stock["symbol"],
        "status": ai_status,
        "reasons": reasons,
        "confidence": confidence
    }
