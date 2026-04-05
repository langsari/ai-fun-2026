import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# 1. โหลดข้อมูลใหม่ 200 ข้อที่คุณเพิ่งแก้ใน CSV
df = pd.read_csv('data/stock_dataset.csv')

# 2. เตรียมข้อมูล (X คือตัวเลข Ratios, y คือคำตอบ Halal/Haram)
X = df[['interest_ratio', 'debt_ratio', 'non_halal_ratio']]
y = df['label'].apply(lambda x: 1 if x == 'Halal' else 0)

# รายชื่อโมเดลที่ต้องสร้างใหม่
models = {
    "randomforest": RandomForestClassifier(n_estimators=100, random_state=42),
    "svm": SVC(probability=True, random_state=42),
    "logisticregression": LogisticRegression(),
    "knn": KNeighborsClassifier(n_neighbors=5),
    "naivebayes": GaussianNB()
}

# 3. วนลูปเทรนและเซฟทับไฟล์เก่าในโฟลเดอร์ model
if not os.path.exists('model'):
    os.makedirs('model')

for name, clf in models.items():
    clf.fit(X, y)
    joblib.dump(clf, f'model/{name}_model.pkl')
    print(f"✅ Trained and Updated: {name}_model.pkl")

print("\n🚀 AI 'Learing' complete! ตอนนี้ AI จำเกณฑ์ใหม่เรียบร้อยแล้วครับ")
