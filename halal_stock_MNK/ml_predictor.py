import joblib
import os
import numpy as np
import pandas as pd
from shariah_rules import check_sector, check_financial_ratios

# --- CONFIGURATION ---
MODEL_NAMES = ["randomforest", "svm", "logisticregression", "knn", "naivebayes"]
MODEL_DIR = "model"
MODELS = {}

# โหลดโมเดลเตรียมไว้ตั้งแต่เริ่ม Start Script
for name in MODEL_NAMES:
    model_path = os.path.join(MODEL_DIR, f"{name}_model.pkl")
    try:
        if os.path.exists(model_path):
            MODELS[name] = joblib.load(model_path)
        else:
            print(f"⚠️ Warning: Model file {model_path} not found.")
            MODELS[name] = None
    except Exception as e:
        print(f"❌ Error loading {name}: {e}")
        MODELS[name] = None

def ai_analyze(stock, sector_name):
    """
    วิเคราะห์หุ้นด้วย AI Consensus และ Shariah Rules
    คืนค่า: (status, confidence, reasons, ai_comparison_list)
    """
    try:
        # 1. คำนวณ Ratios (รองรับทั้งกรณีส่งค่า Ratio มาเลย หรือส่งตัวเลขดิบมา)
        # ถ้ามีค่า ratio มาอยู่แล้วให้ใช้เลย ถ้าไม่มีให้คำนวณจากตัวเลขดิบ
        interest_ratio = stock.get("interest_ratio")
        if interest_ratio is None:
            interest_income = stock.get("interest_income", 0)
            total_income = stock.get("total_income", 1)
            interest_ratio = interest_income / (total_income if total_income > 0 else 1)

        debt_ratio = stock.get("debt_ratio")
        if debt_ratio is None:
            interest_debt = stock.get("interest_debt", 0)
            total_assets = stock.get("total_assets", 1)
            debt_ratio = interest_debt / (total_assets if total_assets > 0 else 1)

        non_halal_ratio = stock.get("non_halal_ratio")
        if non_halal_ratio is None:
            non_halal_income = stock.get("non_halal_income", 0)
            total_income = stock.get("total_income", 1)
            non_halal_ratio = non_halal_income / (total_income if total_income > 0 else 1)

        # เตรียม DataFrame สำหรับ AI (ต้องมีชื่อ Column ตรงกับตอน Train)
        X_df = pd.DataFrame([[interest_ratio, debt_ratio, non_halal_ratio]], 
                            columns=['interest_ratio', 'debt_ratio', 'non_halal_ratio'])
        
    except Exception as e:
        return "Error", 0, [f"Data Processing Error: {str(e)}"], []

    # 2. ตรวจสอบตามกฎ Shariah พื้นฐาน (Rule-based)
    sector_ok, sector_msg = check_sector(sector_name)
    financial_ok, financial_reasons = check_financial_ratios(interest_ratio, debt_ratio, non_halal_ratio)
    
    # 3. รัน AI ทุกตัว
    ai_comparison_list = []
    pass_count = 0
    total_confidence = 0
    valid_models = 0

    for name, model in MODELS.items():
        if model is not None:
            try:
                # ทำนายผล (1=Halal, 0=Haram)
                pred = model.predict(X_df)[0]
                prob_arr = model.predict_proba(X_df)[0]
                
                # แปลงค่าจาก numpy เป็น python float เพื่อป้องกัน Error ใน HTML
                conf = float(np.max(prob_arr))
                status = "HALAL" if pred == 1 else "HARAM"
                
                if pred == 1: pass_count += 1
                total_confidence += conf
                valid_models += 1

                ai_comparison_list.append({
                    "name": name.replace("regression", "").upper(),
                    "status": status,
                    "confidence": round(conf * 100, 2)
                })
            except Exception as e:
                ai_comparison_list.append({"name": name.upper(), "status": "ERROR", "confidence": 0})
        else:
            ai_comparison_list.append({"name": name.upper(), "status": "MISSING", "confidence": 0})

    # 4. สรุปผลการตัดสินใจ (Final Consensus)
    # เงื่อนไข: 
    # - ต้องผ่านด่านธุรกิจ (Sector)
    # - ต้องผ่านด่านการเงิน (Financial Ratios)
    # - AI ส่วนใหญ่ (3 ใน 5) ต้องเห็นพ้องว่า Halal
    
    ai_passed = (pass_count >= 3) if valid_models > 0 else False
    
    if not sector_ok:
        final_status = "HARAM"
        final_reasons = [f"Sector: {sector_msg}"]
    elif not financial_ok:
        final_status = "HARAM"
        final_reasons = [f"Sector: {sector_msg}"] + [f"Financial: {r}" for r in financial_reasons]
    elif not ai_passed:
        final_status = "HARAM"
        final_reasons = [f"Sector: {sector_msg}", "Financial: All ratios within thresholds", f"AI Decision: Only {pass_count}/{valid_models} models passed."]
    else:
        final_status = "HALAL"
        final_reasons = [f"Sector: {sector_msg}", "Financial: All financial ratios are within halal thresholds"]

    # คำนวณความมั่นใจเฉลี่ย
    avg_conf = (total_confidence / valid_models) if valid_models > 0 else 0
    
    return final_status, round(float(avg_conf) * 100, 1), final_reasons, ai_comparison_list
