import logging
from functools import lru_cache
from flask import Flask, render_template, request
import pandas as pd

# นำเข้าโมดูลส่วนตัวของคุณ
from data_fetcher import get_stock_data
from ml_predictor import ai_analyze

# --- CONFIGURATION ---
CSV_PATH = "data/stock_dataset.csv"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# ตั้งค่า Logging
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = "halal-stock-ai-multi-model-secret"

# --- DATA PRE-LOADING ---
def load_initial_dataset():
    """โหลด Dataset เพียงครั้งเดียวตอนเริ่มระบบ"""
    try:
        df = pd.read_csv(CSV_PATH)
        logger.info(f"Successfully loaded {len(df)} records from {CSV_PATH}")
        return df.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Cannot load CSV: {e}")
        return []

DATASET_RECORDS = load_initial_dataset()

# --- CACHING & ANALYSIS ---
@lru_cache(maxsize=128)
def cached_stock_analysis(ticker):
    """
    ฟังก์ชันวิเคราะห์หุ้น: รองรับการรับค่า 4 อย่างจาก ai_analyze 
    (status, confidence, reasons, ai_list)
    """
    stock_data = get_stock_data(ticker)
    if not stock_data:
        return None
    
    sector = stock_data.get("sector", "Unknown")
    
    # ดึงผลการวิเคราะห์ (รับมาเป็น Tuple เพื่อความปลอดภัย)
    analysis_results = ai_analyze(stock_data, sector)
    
    # แยกตัวแปรตามลำดับ (ถ้าส่งมาไม่ครบ 4 โปรแกรมจะไม่พังเพราะใช้การเช็ค Index)
    status = analysis_results[0] if len(analysis_results) > 0 else "Error"
    confidence = analysis_results[1] if len(analysis_results) > 1 else 0.0
    reasons = analysis_results[2] if len(analysis_results) > 2 else []
    ai_list = analysis_results[3] if len(analysis_results) > 3 else []

    return {
        "ticker": ticker,
        "sector": sector,
        "status": status,
        "confidence": round(float(confidence) * 100, 2),
        "reasons": reasons,
        "ai_list": ai_list  # รายชื่อโมเดล AI ทั้งหมดเพื่อนำไปทำตารางเปรียบเทียบ
    }

# --- ROUTES ---
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        raw_ticker = request.form.get("ticker", "")
        ticker = raw_ticker.upper().strip()
        
        if not ticker:
            error = "กรุณาระบุชื่อหุ้นก่อนทำการวิเคราะห์"
        else:
            try:
                logger.info(f"Analyzing: {ticker}")
                result = cached_stock_analysis(ticker)
                
                if not result:
                    error = f"ไม่พบข้อมูลของหุ้น '{ticker}'"
                    
            except Exception as e:
                logger.error(f"Error analyzing {ticker}: {str(e)}")
                error = f"เกิดข้อผิดพลาด: {str(e)}"

    return render_template(
        "index.html",
        result=result,
        error=error,
        dataset=DATASET_RECORDS
    )

if __name__ == "__main__":
    app.run(debug=True, port=5000)
