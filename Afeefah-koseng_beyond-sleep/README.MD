
| Column            | ความหมาย                   |
| ----------------- | -------------------------- |
| Age               | อายุ                       |
| Gender            | เพศ                        |
| Occupation        | อาชีพ                      |
| Sleep_Duration    | ชั่วโมงการนอน              |
| Stress_Level      | ระดับความเครียด (1-10)     |
| BMI_Category      | น้ำหนัก                    |
| Physical_Activity | ออกกำลังกาย (minutes/day)  |
| Screen_Time       | ใช้มือถือก่อนนอน (minutes) |
| Sleep_Quality     | คุณภาพการนอน (1-10)        |
| Sleep_Disorder    | ปัญหาการนอน                |


Sleep_Disorder จะมีค่าเช่น
None
Insomnia
Sleep_Apnea(ภาวะหยุดหายใจขณะหลับ)

## 🧠 Insomnia Detection & Sound Therapy Recommendation

## 📌 Project Description
This project uses Machine Learning to predict insomnia risk based on user lifestyle data such as sleep duration, stress level, physical activity, and screen time.

It also recommends sound therapy (e.g., rain sound, white noise, delta waves, pink_noise, ocean_wave) to help improve sleep quality.

## ✨ Features
- Predict sleep condition (Normal / Insomnia)
- Recommend sound therapy
- Play relaxing sounds directly
- Simple and interactive user input

## 📊 Dataset
The dataset includes:
- Age
- Gender
- Sleep Duration
- Stress Level
- Physical Activity
- Screen Time
- Sleep Disorder

The data is used to train a Decision Tree model.

## 🤖 Model
We used a Decision Tree Classifier with max_depth=3 to avoid overfitting and improve interpretability.

## 🚀 How to Run

1. Open the notebook
2. Run all cells
3. Enter user data:
   - Age
   - Gender
   - Sleep Duration
   - Stress Level
   - Physical Activity
   - Screen Time
4. Get prediction and recommended sound

## 📈 Example Output

Sleep Condition: Normal  
Recommended Sound Therapy: White Noise  
Playing White Noise...

## 🛠 Technologies
- Python
- Pandas
- Scikit-learn
- Matplotlib
- IPython Display

## ⚙️ Environment Setup

1. Clone the repository:
   git clone https://github.com/your-repo/ai-fun-2026.git

2. Navigate to the project folder:
   cd Afeefah-koseng_beyond-sleep

3. Install required libraries:
   pip install pandas numpy scikit-learn matplotlib

4. Run the notebook:
   Open insomnia_ai_project.ipynb and run all cells

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

## 🧠 Problem Statement
The goal of this project is to predict whether a person is at risk of insomnia based on lifestyle factors and recommend suitable sound therapy.

## 📊 Data Collection
The dataset contains information about sleep habits, stress levels, and lifestyle factors.

## 👩‍💻 Author
Afeefah Koseng