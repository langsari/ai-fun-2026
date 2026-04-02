"""
calorie_calculator.py
---------------------
โมดูลหลักสำหรับค้นหาและคำนวณแคลอรี่อาหารไทย
"""

import pandas as pd
import os

# ─────────────────────────────────────────────
# โหลดฐานข้อมูล
# ─────────────────────────────────────────────

def load_database(csv_path: str = None) -> pd.DataFrame:
    """โหลดไฟล์ CSV ฐานข้อมูลอาหาร"""
    if csv_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_dir, "..", "data", "thai_foods.csv")
    
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    return df


# ─────────────────────────────────────────────
# ค้นหาอาหาร
# ─────────────────────────────────────────────

def search_food(query: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    ค้นหาอาหารจากชื่อไทยหรืออังกฤษ (case-insensitive, partial match)
    
    Parameters:
        query : ชื่ออาหารที่ต้องการค้นหา
        df    : DataFrame ฐานข้อมูลอาหาร
    
    Returns:
        DataFrame ที่ตรงกับคำค้นหา
    """
    query = query.strip().lower()
    mask = (
        df["food_name_th"].str.lower().str.contains(query, na=False) |
        df["food_name_en"].str.lower().str.contains(query, na=False)
    )
    return df[mask].reset_index(drop=True)


# ─────────────────────────────────────────────
# คำนวณแคลอรี่
# ─────────────────────────────────────────────

def calculate_calories(food_row: pd.Series, quantity: float = 1.0) -> dict:
    """
    คำนวณสารอาหารตามจำนวนที่กิน
    
    Parameters:
        food_row : แถวข้อมูลอาหาร 1 รายการ
        quantity : จำนวนหน่วยที่กิน (default = 1.0)
    
    Returns:
        dict ข้อมูลสารอาหารทั้งหมด
    """
    return {
        "ชื่ออาหาร"   : food_row["food_name_th"],
        "Food (EN)"  : food_row["food_name_en"],
        "หมวดหมู่"   : food_row["category"],
        "จำนวน"      : f"{quantity} {food_row['serving_unit']}",
        "น้ำหนัก (g)": round(food_row["serving_size_g"] * quantity, 1),
        "แคลอรี่"    : round(food_row["calories"]   * quantity, 1),
        "โปรตีน (g)" : round(food_row["protein_g"]  * quantity, 1),
        "คาร์บ (g)"  : round(food_row["carbs_g"]    * quantity, 1),
        "ไขมัน (g)"  : round(food_row["fat_g"]      * quantity, 1),
        "ใยอาหาร (g)": round(food_row["fiber_g"]    * quantity, 1),
    }


# ─────────────────────────────────────────────
# สรุปมื้ออาหาร
# ─────────────────────────────────────────────

def summarize_meal(meal_items: list[dict]) -> dict:
    """
    รวมสารอาหารของทุกรายการในมื้อนั้น
    
    Parameters:
        meal_items : list ของ dict จาก calculate_calories()
    
    Returns:
        dict สรุปสารอาหารรวม
    """
    if not meal_items:
        return {}

    total = {
        "รายการอาหาร" : [item["ชื่ออาหาร"] for item in meal_items],
        "แคลอรี่รวม"  : round(sum(item["แคลอรี่"]     for item in meal_items), 1),
        "โปรตีนรวม (g)": round(sum(item["โปรตีน (g)"] for item in meal_items), 1),
        "คาร์บรวม (g)" : round(sum(item["คาร์บ (g)"]  for item in meal_items), 1),
        "ไขมันรวม (g)" : round(sum(item["ไขมัน (g)"]  for item in meal_items), 1),
        "ใยอาหารรวม (g)": round(sum(item["ใยอาหาร (g)"] for item in meal_items), 1),
    }
    return total


# ─────────────────────────────────────────────
# แนะนำพลังงานต่อวัน (TDEE ขั้นพื้นฐาน)
# ─────────────────────────────────────────────

ACTIVITY_MULTIPLIER = {
    "นั่งทำงาน (sedentary)"      : 1.2,
    "ออกกำลังเบา (light)"        : 1.375,
    "ออกกำลังปานกลาง (moderate)" : 1.55,
    "ออกกำลังหนัก (active)"      : 1.725,
    "ออกกำลังหนักมาก (very active)": 1.9,
}

def calculate_bmr(weight_kg: float, height_cm: float, age: int, gender: str) -> float:
    """
    คำนวณ BMR (Basal Metabolic Rate) ด้วยสูตร Mifflin-St Jeor
    
    Parameters:
        weight_kg : น้ำหนัก (กิโลกรัม)
        height_cm : ส่วนสูง (เซนติเมตร)
        age       : อายุ (ปี)
        gender    : 'male' หรือ 'female'
    
    Returns:
        ค่า BMR (kcal/วัน)
    """
    if gender.lower() in ("male", "ชาย"):
        return (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5
    else:
        return (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161


def calculate_tdee(bmr: float, activity_level: str) -> float:
    """คำนวณ TDEE จาก BMR และระดับกิจกรรม"""
    multiplier = ACTIVITY_MULTIPLIER.get(activity_level, 1.2)
    return round(bmr * multiplier, 1)
