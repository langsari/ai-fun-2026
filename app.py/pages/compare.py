import streamlit as st

st.set_page_config(page_title="Compare Cars", layout="wide")

# ==========================================
# 📥 1. ดึงข้อมูลจากลิสต์ (Session State)
# ==========================================
# ตรวจสอบว่ามีข้อมูลจากหน้า Home ส่งมาหรือไม่
compare_data = st.session_state.get("compare_list", [])

# กำหนดค่าตั้งต้นสำหรับคันที่ 1 (ถ้าในลิสต์มีข้อมูลตัวที่ 1)
if len(compare_data) >= 1:
    c1_default_name = compare_data[0]["name"]
    c1_default_price = compare_data[0]["price"]
else:
    c1_default_name = "รถคันที่ 1"
    c1_default_price = 0

# กำหนดค่าตั้งต้นสำหรับคันที่ 2 (ถ้าในลิสต์มีข้อมูลตัวที่ 2)
if len(compare_data) >= 2:
    c2_default_name = compare_data[1]["name"]
    c2_default_price = compare_data[1]["price"]
else:
    c2_default_name = "รถคันที่ 2"
    c2_default_price = 0

# ==========================================
# 🎨 2. ปรับแต่ง CSS
# ==========================================
st.markdown("""
<style>
    .stMetric { background-color: #1c1f26; padding: 20px; border-radius: 12px; border: 1px solid #30363d; }
    .analysis-box { background-color: #23272e; padding: 20px; border-radius: 10px; border-left: 5px solid #58a6ff; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Compare Cars")

# แจ้งเตือนหากเลือกไม่ครบ
if len(compare_data) < 2:
    st.warning(f"⚠️ เลือกไปแล้ว {len(compare_data)} คัน (แนะนำให้เลือกจากหน้า Home ให้ครบ 2 คัน)")

st.divider()

# ==========================================
# 📥 3. ส่วนรับข้อมูล (ดึงค่ามาใส่ในช่องอัตโนมัติ)
# ==========================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("🍎 คันที่ 1")
    name1 = st.text_input("ชื่อรุ่นรถ (1)", c1_default_name, key="n1")
    val1 = st.number_input(f"ราคาของ {name1}", 0, 5000000, int(c1_default_price), key="v1")

with col2:
    st.subheader("🚙 คันที่ 2")
    name2 = st.text_input("ชื่อรุ่นรถ (2)", c2_default_name, key="n2")
    val2 = st.number_input(f"ราคาของ {name2}", 0, 5000000, int(c2_default_price), key="v2")

st.write(" ")

# ปุ่มคำนวณ
if st.button("🚀 เริ่มการเปรียบเทียบเชิงลึก", use_container_width=True):
    diff = abs(val1 - val2)
    max_val = max(val1, val2, 1)
    
    # แสดงผล Metric
    m1, m2 = st.columns(2)
    with m1:
        st.metric(label=f"ข้อมูลของ {name1}", value=f"{val1:,}", 
                  delta=f"-{diff:,}" if val1 < val2 else (f"+{diff:,}" if val1 > val2 else "เท่ากัน"),
                  delta_color="normal" if val1 < val2 else "inverse")
        st.progress(val1 / max_val)
        
    with m2:
        st.metric(label=f"ข้อมูลของ {name2}", value=f"{val2:,}", 
                  delta=f"-{diff:,}" if val2 < val1 else (f"+{diff:,}" if val2 > val1 else "เท่ากัน"),
                  delta_color="normal" if val2 < val1 else "inverse")
        st.progress(val2 / max_val)

    st.divider()

    # บทวิเคราะห์
    res_l, res_r = st.columns([1, 1.5])
    with res_l:
        st.subheader("💡 ผลการวิเคราะห์")
        if val1 < val2:
            st.success(f"**{name1}** ประหยัดกว่า **{diff:,}** บาท")
        elif val2 < val1:
            st.success(f"**{name2}** ประหยัดกว่า **{diff:,}** บาท")
        else:
            st.info("ทั้งสองคันราคาเท่ากัน")

    with res_r:
        st.markdown(f"""
        <div class="analysis-box">
            <h4>📝 สรุปภาพรวม</h4>
            <ul>
                <li>ส่วนต่างราคา: <b>{diff:,} บาท</b></li>
                <li>คันที่ราคาดีที่สุด: <b>{name1 if val1 <= val2 else name2}</b></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ปุ่มล้างข้อมูล
if st.sidebar.button("🗑️ ล้างข้อมูลเปรียบเทียบ"):
    st.session_state["compare_list"] = []
    st.rerun()
