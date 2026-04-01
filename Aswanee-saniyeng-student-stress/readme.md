
````markdown
# 🎓 AI System for Student Stress Level Assessment

> A simple AI and Machine Learning project to predict student stress levels.

---

## 📌 About This Project
This project was created for the **AI-FUN-2026** course.

The goal of this project is to build a machine learning system that can predict a student’s **stress level** based on their daily habits, academic pressure, and mental condition.

The system predicts stress into 3 levels:

- 🟢 Low
- 🟡 Moderate
- 🔴 High

---

## 🎯 Project Objective
The objectives of this project are to:

- study the factors that affect student stress
- build a machine learning model
- predict student stress level automatically
- create a simple prediction system in Jupyter Notebook

---

## 🧠 Problem Statement
Many students experience stress because of:

- too much studying
- lack of sleep
- academic pressure
- social media use
- exam anxiety
- mental fatigue

This project uses **AI and Machine Learning** to help predict stress levels based on these factors.

---

## 📊 Dataset
This project uses a dataset of **50 students**.

### Features used:
- Age
- Gender
- Study_Hours
- Sleep_Hours
- Academic_Pressure
- Social_Media_Hours
- Physical_Activity
- Exam_Anxiety
- Mental_Fatigue

### Target:
- Stress_Level

---

## ⚙️ Tools and Libraries
This project was built using:

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- ipywidgets

---

## 🔄 Machine Learning Workflow
This project follows these steps:

1. Load the dataset  
2. Explore and understand the data  
3. Prepare the data  
4. Train machine learning models  
5. Evaluate the models  
6. Select the best model  
7. Predict student stress level  

---

## 🤖 Models Used
The following machine learning models were tested:

- Logistic Regression
- Decision Tree
- Random Forest

### 🏆 Best Model:
**Random Forest Classifier**

This model gave the best performance for this project.

---

## 🛠️ Data Preprocessing
Before training the model, the data was prepared by:

- converting gender into numbers
- separating features and target
- scaling the data using StandardScaler
- splitting data into training and testing sets

---

## 📉 Model Evaluation
The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### Example Result
- Accuracy: 0.87

> Replace this with your actual result from Jupyter Notebook.

---

## 🖥️ Prediction System
This project also includes a simple **interactive prediction form** in Jupyter Notebook using **ipywidgets**.

Users can enter student information such as:

- age
- gender
- study hours
- sleep hours
- academic pressure
- social media hours
- physical activity
- exam anxiety
- mental fatigue

Then the system predicts the stress level.

---

## 🔍 Example Prediction

### Input
```python
new_student = {
    "Age": 21,
    "Gender": "Female",
    "Study_Hours": 9,
    "Sleep_Hours": 4.5,
    "Academic_Pressure": 9,
    "Social_Media_Hours": 6,
    "Physical_Activity": 1,
    "Exam_Anxiety": 9,
    "Mental_Fatigue": 9
}
````

### Output

```text
High
```

---

## 📂 Project Structure

```bash
AI-FUN-2026/
│
├── data/
│   └── student_stress.csv
│
├── notebooks/
│   └── student_stress_project.ipynb
│
├── models/
│   ├── stress_model.pkl
│   └── scaler.pkl
│
├── README.md
└── requirements.txt
```

---

## ▶️ How to Run

### 1. Install required libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib ipywidgets
```

### 2. Open Jupyter Notebook

```bash
jupyter notebook
```

### 3. Open the notebook file

```bash
student_stress_project.ipynb
```

### 4. Run all cells

---

## 💾 Saved Files

This project saves the trained model for future use:

```bash
models/stress_model.pkl
models/scaler.pkl
```

---

## 📌 Conclusion

This project shows that machine learning can be used to predict student stress levels.

It helps demonstrate how AI can be applied to solve a real student-related problem in a simple and practical way.

---

## 🔮 Future Improvement

This project can be improved by:

* using a bigger dataset
* adding more student behavior features
* building a web app version
* improving model accuracy

---

## 👨‍💻 Author

**Name:** Aswanee Saniyeng
**Project:** AI System for Student Stress Level Assessment
**Course:** AI-FUN-2026

````

