import streamlit as st

st.set_page_config(page_title="Car AI Advisor", page_icon="🚗", layout="wide")

# ส่วน Sidebar
with st.sidebar:
    st.title("🚗 Main Menu")
    if st.button("🗑️ ล้างข้อมูลการเปรียบเทียบ", use_container_width=True):
        st.session_state["compare_list"] = []
        st.success("ล้างข้อมูลแล้ว")

# หน้า Landing
st.markdown("# 🚗 Car AI Advisor")
st.markdown("### ระบบแนะนำรถยนต์และเปรียบเทียบความคุ้มค่า")
st.divider()

col1, col2 = st.columns(2)
with col1:
    if st.button("🏠 ไปหน้าค้นหารถ (Home)", use_container_width=True):
        st.switch_page("pages/home.py")
with col2:
    if st.button("📊 ไปหน้าเปรียบเทียบ (Compare)", use_container_width=True):
        st.switch_page("pages/compare.py")

st.info("💡 วิธีใช้: ไปที่หน้า Home เพื่อค้นหารถที่ชอบ แล้วกดปุ่ม 'เปรียบเทียบ' ให้ครบ 2 คัน ระบบจะพามาหน้าเปรียบเทียบโดยอัตโนมัติ")
