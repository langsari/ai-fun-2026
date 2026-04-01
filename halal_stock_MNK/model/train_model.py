import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# โหลดข้อมูล training
df = pd.read_csv("data/training_data.csv")

# features
X = df[["interest_ratio", "debt_ratio", "non_halal_ratio"]]

# label
y = df["label"]

# สร้างโมเดล
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# สร้างโฟลเดอร์ model ถ้ายังไม่มี
os.makedirs("model", exist_ok=True)

# บันทึกโมเดล
joblib.dump(model, "model/halal_model.pkl")

print("✅ เทรนโมเดลเสร็จแล้ว")
