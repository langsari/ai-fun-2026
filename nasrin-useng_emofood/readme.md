# 🍜 Emotional Food AI

A Machine Learning project that recommends food based on a person's **emotion** and the current **weather condition**.

---

## Project Overview

People often choose food based on their mood and the weather around them. This project solves the problem of choosing the right food for comfort or satisfaction by using a **Machine Learning model (Random Forest Classifier)** trained on a dataset combining emotions, weather types, and suitable food items.

- **Input features:** `emotion`, `weather`
- **Output:** `food` recommendation

---

## Dataset

The dataset (`emotional_food_1000.csv`) contains **999 samples** with three columns:

| Column    | Description                                      | Example Values                          |
|-----------|--------------------------------------------------|-----------------------------------------|
| `emotion` | The user's current emotional state               | happy, sad, stressed, relaxed, angry, tired, bored, excited |
| `weather` | The current weather condition                    | rainy, sunny, cold, cloudy, hot         |
| `food`    | The recommended food for that emotion & weather  | Fried Chicken, Tom Yum Soup, Pizza, ... |

- **Total records:** 999
- **Unique emotions:** 8
- **Unique weather types:** 5
- **Unique food items:** 16
- **Most frequent emotion:** sad (135 samples)
- **Most frequent weather:** rainy (217 samples)
- **Most recommended food:** Fried Chicken (184 samples)

---

## Project Structure

```
emotional-food-ai/
├── data/
│   └── emotional_food_1000.csv
└── notebook/
    └── emotional_food_ai.ipynb
```

---

## Workflow

### 1. Load Dataset
```python
import pandas as pd
df = pd.read_csv("../data/emotional_food_1000.csv")
df.head()
```

### 2. Explore Dataset
```python
df.info()
df.describe(include='all')
df['emotion'].value_counts()
```

### 3. Visualize Dataset
```python
import matplotlib.pyplot as plt

# Bar plot for weather distribution
df['weather'].value_counts().plot(kind='bar')
plt.title("Weather Distribution")
plt.show()

# Pie chart for emotion distribution
df['emotion'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title("Emotion Distribution")
plt.show()
```

### 4. Data Preprocessing & Train/Test Split
```python
from sklearn.preprocessing import LabelEncoder

le_emotion = LabelEncoder()
le_weather = LabelEncoder()
le_food = LabelEncoder()

df['emotion'] = le_emotion.fit_transform(df['emotion'])
df['weather'] = le_weather.fit_transform(df['weather'])

X = df[['emotion', 'weather']]
y = le_food.fit_transform(df['food'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 5. Train Model
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X, y)
```

### 6. Prediction with User Input
```python
print("=== Emotion Menu ===")
print("0 = happy, 1 = sad, 2 = stressed, 3 = bored, 4 = angry, 5 = relaxed")

print("\n=== Weather Menu ===")
print("0 = hot, 1 = rainy, 2 = cold, 3 = cloudy, 4 = windy")

emotion_num = int(input("\nEnter emotion number: "))
weather_num = int(input("Enter weather number: "))

input_data = pd.DataFrame([[emotion_num, weather_num]], columns=['emotion', 'weather'])
pred = model.predict(input_data)
result = le_food.inverse_transform(pred)

print("\nRecommended Food:", result[0])
```

---

## Emotion & Weather Encoding

| Code | Emotion   |
|------|-----------|
| 0    | happy     |
| 1    | sad       |
| 2    | stressed  |
| 3    | bored     |
| 4    | angry     |
| 5    | relaxed   |

| Code | Weather |
|------|---------|
| 0    | hot     |
| 1    | rainy   |
| 2    | cold    |
| 3    | cloudy  |
| 4    | windy   |

---

## Example Output

```
=== Emotion Menu ===
0 = happy, 1 = sad, 2 = stressed, 3 = bored, 4 = angry, 5 = relaxed

=== Weather Menu ===
0 = hot, 1 = rainy, 2 = cold, 3 = cloudy, 4 = windy

Recommended Food: Noodles
```

---

## Requirements

```
pandas
matplotlib
scikit-learn
```

Install with:
```bash
pip install pandas matplotlib scikit-learn
```

---

## Technologies Used

- **Python**
- **Pandas** – Data loading and exploration
- **Matplotlib** – Data visualization
- **Scikit-learn** – LabelEncoder, train_test_split, RandomForestClassifier

---

## Author
**Name:** Nasrin Useng
