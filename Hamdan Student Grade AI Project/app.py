import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# -----------------------------
# CONFIG + DARK STYLE
# -----------------------------
st.set_page_config(page_title="AI Predictor", layout="centered")

st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
}
.card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 0 10px rgba(0,0,0,0.5);
    margin-bottom: 20px;
    text-align: center;
}
.big-text {
    font-size: 40px;
    font-weight: bold;
}
.fade-in {
    animation: fadeIn 1s ease-in;
}
@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}
</style>
""", unsafe_allow_html=True)

st.title("🎓 AI Student Risk Predictor")

# -----------------------------
# LOAD MODEL
# -----------------------------
model = tf.keras.models.load_model("model.keras")
scaler = joblib.load("scaler.pkl")

features = [
    "age","Medu","Fedu","traveltime","studytime","failures",
    "famrel","freetime","goout","Dalc","Walc","health",
    "absences","G1","G2"
]

# -----------------------------
# INPUT UI (CARD)
# -----------------------------
st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
st.subheader("📝 Enter Student Data")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 15, 22, 17)
    Medu = st.slider("Mother Education", 0, 4, 2)
    Fedu = st.slider("Father Education", 0, 4, 2)
    traveltime = st.slider("Travel Time", 1, 4, 2)
    studytime = st.slider("Study Time", 1, 4, 2)
    failures = st.slider("Failures", 0, 3, 0)
    famrel = st.slider("Family Relationship", 1, 5, 3)

with col2:
    freetime = st.slider("Free Time", 1, 5, 3)
    goout = st.slider("Go Out", 1, 5, 3)
    Dalc = st.slider("Workday Alcohol", 1, 5, 1)
    Walc = st.slider("Weekend Alcohol", 1, 5, 2)
    health = st.slider("Health", 1, 5, 3)
    absences = st.slider("Absences", 0, 30, 5)
    G1 = st.slider("First Grade", 0, 20, 10)
    G2 = st.slider("Second Grade", 0, 20, 10)

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# PREDICT BUTTON
# -----------------------------
if st.button("🚀 Predict"):

    input_data = pd.DataFrame([[
        age, Medu, Fedu, traveltime, studytime, failures,
        famrel, freetime, goout, Dalc, Walc, health,
        absences, G1, G2
    ]], columns=features)

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0][0]
    prediction = max(0, min(20, prediction))

    # -----------------------------
    # RESULT CARD
    # -----------------------------
    if prediction < 10:
        color = "red"
        text = "🔴 HIGH RISK"
    elif prediction < 14:
        color = "orange"
        text = "🟠 MEDIUM RISK"
    else:
        color = "green"
        text = "🟢 LOW RISK"

    st.markdown(f"""
    <div class="card fade-in">
        <div class="big-text" style="color:{color};">{text}</div>
        <h3>Predicted Score: {prediction:.2f}</h3>
    </div>
    """, unsafe_allow_html=True)