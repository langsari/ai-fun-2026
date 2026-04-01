
# 🧠 Insomnia Detection & Sound Therapy Recommendation

---

## 📌 Project Description
This project uses Machine Learning to predict whether a person is at risk of insomnia based on lifestyle factors such as sleep duration, stress level, physical activity, and screen time.

Additionally, the system recommends appropriate sound therapy (e.g., rain sound, white noise, delta waves) to help improve sleep quality.

---

## ⚙️ 1. Environment Setup Instructions

1. Clone the repository:
   git clone https://github.com/langsari/ai-fun-2026.git

2. Navigate to the project folder:
   cd Afeefah-koseng_beyond-sleep

3. Install required libraries:
   pip install pandas numpy scikit-learn matplotlib

4. Run the notebook:
   Open insomnia_ai_project.ipynb and run all cells

---

## 🧠 2.1 Look at the Big Picture (Problem Statement)

The goal of this project is to:
- Predict sleep condition (Normal or Insomnia)
- Recommend suitable sound therapy based on user behavior

This helps users improve their sleep quality using AI.

---

## 📊 2.2 Get the Data

The dataset contains the following features:

| Column            | Description                  |
|------------------|-----------------------------|
| Age              | อายุ                         |
| Gender           | เพศ                          |
| Occupation       | อาชีพ                        |
| Sleep_Duration   | ชั่วโมงการนอน                |
| Stress_Level     | ระดับความเครียด (1-10)       |
| BMI_Category     | หมวดหมู่น้ำหนัก             |
| Physical_Activity| การออกกำลังกาย              |
| Screen_Time      | ใช้มือถือก่อนนอน             |
| Sleep_Quality    | คุณภาพการนอน (1-10)          |
| Sleep_Disorder   | ปัญหาการนอน                 |

Target variable:
- Sleep_Disorder (None / Insomnia / Sleep Apnea)

---

## 📈 2.3 Explore and Visualize Data

We explored the dataset using:
- `df.describe()` for summary statistics
- `df["Sleep_Disorder"].value_counts()` for class distribution

This helps understand patterns and data balance.

---

## 🧹 2.4 Prepare the Data

- Converted categorical data (Gender → numeric)
- Selected relevant features:
  - Age
  - Gender
  - Sleep_Duration
  - Stress_Level
  - Physical_Activity
  - Screen_Time
- Transformed target variable into binary:
  - Insomnia = 1
  - Normal = 0

---

## 🤖 2.5 Select Model and Train

We used:
- Decision Tree Classifier

Configuration:
- `max_depth = 3`

Data split:
- 80% training
- 20% testing

---

## ⚙️ 2.6 Fine-tune the Model

We limited tree depth to avoid overfitting and improve generalization.

---

## 🎤 2.7 Present the Solution

The system:
- Takes user input
- Predicts sleep condition
- Recommends sound therapy
- Plays relaxing audio

---

## 🚀 2.8 Launch, Monitor, Maintain

Future improvements:
- Improve model accuracy
- Add more sound types
- Build a web or mobile application

---

## 📁 Project Structure

```text
Afeefah-koseng_beyond-sleep/
│
├── data/
│   └── insomnia_sleep_dataset.csv
│
├── sounds/
│   ├── rain.mp3
│   ├── white_noise.mp3
│   ├── delta_wave.mp3
│   ├── pink_noise.mp3
│   └── ocean_wave.mp3
│
├── insomnia_ai_project.ipynb
└── README.md
```

---

## 🤖 Model

- Algorithm: Decision Tree Classifier
- Depth: 3

Features:
- Age
- Gender
- Sleep Duration
- Stress Level
- Physical Activity
- Screen Time

---

## 📊 Model Evaluation

The model was evaluated using accuracy score.

Example:
Model Accuracy: ~0.80 - 0.90

---

## 🚀 How to Use

1. Run all cells
2. Enter user data
3. View prediction
4. Listen to sound

---

## 🎧 Sound Therapy Included

- Rain Sound (Relaxation)
- White Noise (Focus & reduce distractions)
- Delta Wave (Deep sleep)
- Pink Noise (Stable sleep)
- Ocean Wave (Stress relief)

---

## 🛠 Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- IPython Display  

---

## 👩‍💻 Author

Afeefah Koseng

---

## 🎯 Future Improvements

- Add GUI / Web interface
- Improve model accuracy
- Use real-world data




