# 🥗 Thai Calorie Calculator

โปรแกรมคำนวณแคลอรี่อาหารไทย พัฒนาด้วย Python ใช้งานผ่าน Jupyter Notebook

---

## 📁 โครงสร้างโปรเจค

```
calorie-calculator/
├── data/
│   └── thai_foods.csv          # ฐานข้อมูลอาหารไทย 45+ รายการ
├── notebooks/
│   └── calorie_calculator.ipynb  # Jupyter Notebook หลัก
├── src/
│   └── calorie_calculator.py   # โมดูล Python (logic)
├── requirements.txt
└── README.md
```

---

## 🚀 วิธีติดตั้งและใช้งาน

### 1. Clone โปรเจค
```bash
git clone https://github.com/<your-username>/calorie-calculator.git
cd calorie-calculator
```

### 2. ติดตั้ง dependencies
```bash
pip install -r requirements.txt
```

### 3. เปิด Jupyter Notebook
```bash
# เปิดด้วย Anaconda Navigator หรือรันคำสั่ง:
jupyter notebook notebooks/calorie_calculator.ipynb
```

---

## 🍽️ ฟีเจอร์หลัก

| ส่วนที่ | ฟีเจอร์ |
|--------|---------|
| 1 | ดูฐานข้อมูลอาหารทั้งหมดและหมวดหมู่ |
| 2 | ค้นหาอาหารด้วยชื่อไทย/อังกฤษ |
| 3 | บันทึกมื้ออาหารและคำนวณแคลอรี่รวม |
| 4 | กราฟ Macronutrient Breakdown |
| 5 | คำนวณ BMR / TDEE ส่วนตัว |
| 6 | เปรียบเทียบแคลอรี่ในหมวดหมู่เดียวกัน |

---

## 📊 ข้อมูลในฐานข้อมูล

แต่ละรายการมีข้อมูล:
- ชื่ออาหาร (ไทย / อังกฤษ)
- หมวดหมู่ (8 หมวด)
- หน่วยบริโภคและน้ำหนัก (กรัม)
- แคลอรี่ (kcal)
- สารอาหาร: โปรตีน, คาร์บ, ไขมัน, ใยอาหาร

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **pandas** — จัดการและค้นหาข้อมูล
- **matplotlib** — สร้างกราฟ
- **Jupyter Notebook** — UI หลัก

---

## ➕ เพิ่มอาหารเอง

เปิดไฟล์ `data/thai_foods.csv` แล้วเพิ่มแถวตามรูปแบบ:

```
food_name_th,food_name_en,category,serving_unit,serving_size_g,calories,protein_g,carbs_g,fat_g,fiber_g
ข้าวต้มหมู,Rice Porridge with Pork,ข้าวและแป้ง,ชาม,400,280,15.0,42.0,5.0,1.0
```
