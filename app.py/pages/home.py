import streamlit as st
import pandas as pd

# 1. ตั้งค่าหน้ากระดาษ
st.set_page_config(page_title="Car AI Advisor - Home", layout="wide")

# 2. ปรับแต่ง CSS Style
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: white; }
    .card {
        background-color: #1c1f26;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #30363d;
        margin-bottom: 10px;
        min-height: 180px;
    }
    h4 { color: #58a6ff; margin-bottom: 10px; }
    .price-text { color: #ff9f43; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# 3. ฟังก์ชันคำนวณเงินผ่อน
def calculate_monthly(price):
    try:
        down_payment = price * 0.20
        loan_amount = price - down_payment
        total_debt = loan_amount * 1.15 
        return int(total_debt / 60)
    except:
        return 0

# 4. โหลดข้อมูล
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("cars_dataset.csv")
        df["Price (THB)"] = df["Price (THB)"].astype(str).str.replace(",", "")
        df["Price (THB)"] = pd.to_numeric(df["Price (THB)"], errors="coerce")
        return df.dropna(subset=["Price (THB)"])
    except FileNotFoundError:
        return pd.DataFrame()

df = load_data()

# 5. Hero Section
st.markdown("<h1 style='text-align:center;'>🚗 Car AI Advisor</h1>", unsafe_allow_html=True)
st.divider()

# แสดงจำนวนรถที่เลือกไว้ใน Sidebar (เพื่อความสะดวก)
if "compare_list" not in st.session_state:
    st.session_state["compare_list"] = []

with st.sidebar:
    st.write(f"คันที่เลือกไว้: {len(st.session_state['compare_list'])} / 2")
    if st.button("🗑️ ล้างข้อมูลที่เลือก"):
        st.session_state["compare_list"] = []
        st.rerun()

if not df.empty:
    # 6. Input Section
    col1, col2, col3 = st.columns(3)
    with col1:
        budget = st.number_input("งบประมาณสูงสุด (บาท)", 300000, 5000000, 700000, step=50000)
    with col2:
        car_type = st.selectbox("เลือกประเภทรถ", df["Type"].unique())
    with col3:
        distance = st.number_input("ระยะทางที่ขับต่อวัน (กม.)", 1, 500, 30)

    if st.button("🔮 เริ่มการวิเคราะห์", use_container_width=True):
        st.session_state['show_results'] = True

    if st.session_state.get('show_results'):
        mask = (df["Type"] == car_type) & (df["Price (THB)"] <= budget)
        filtered = df[mask].sort_values(by="Price (THB)").head(6)

        if filtered.empty:
            st.warning(f"ไม่พบรถในงบประมาณที่กำหนด")
        else:
            st.subheader("✨ รถที่แนะนำสำหรับคุณ")
            cols = st.columns(3)
            for i, (_, car) in enumerate(filtered.iterrows()):
                with cols[i % 3]:
                    img = car.get("Image_URL", "")
                    if pd.notnull(img) and img != "":
                        st.image(img, use_container_width=True)
                    
                    monthly_pay = calculate_monthly(car["Price (THB)"])
                    
                    st.markdown(f"""
                    <div class="card">
                        <h4>{car['Brand']} {car['Model']}</h4>
                        <span class="price-text">💰 ราคา: {int(car['Price (THB)']):,} บาท</span><br>
                        📊 ผ่อนเริ่มต้น: ~{monthly_pay:,} บาท/เดือน
                    </div>
                    """, unsafe_allow_html=True)

                    # ==========================================
                    # ส่วนแก้ไข: ระบบเลือก 2 คัน
                    # ==========================================
                    if st.button(f"➕ เลือกเปรียบเทียบ {car['Model']}", key=f"btn_{i}", use_container_width=True):
                        new_car = {"name": f"{car['Brand']} {car['Model']}", "price": car["Price (THB)"]}
                        
                        # ตรวจสอบไม่ให้ซ้ำ
                        if new_car not in st.session_state["compare_list"]:
                            if len(st.session_state["compare_list"]) < 2:
                                st.session_state["compare_list"].append(new_car)
                                st.toast(f"เพิ่ม {car['Model']} แล้ว!")
                            else:
                                st.error("เลือกได้สูงสุด 2 คันเท่านั้น ล้างข้อมูลก่อนเลือกใหม่")
                        
                        # ถ้าครบ 2 คัน ให้เด้งไปหน้า Compare ทันที
                        if len(st.session_state["compare_list"]) == 2:
                            st.switch_page("pages/compare.py")
else:
    st.info("กรุณาเตรียมไฟล์ cars_dataset.csv")
