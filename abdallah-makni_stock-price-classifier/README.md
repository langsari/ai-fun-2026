 ## Stock Price Movement Classification  

### Student Information  
Name: Abdallah Makni  
Student ID: 671431030  
Course: Artificial Intelligence Fundamental  
Year: 2026  

---

## 1. Project Overview  

This project is part of the AI-FUN-2026 repository.  

The objective of this project is to build a Machine Learning classification model that predicts the next-day movement of the S&P 500 index (Up or Down) using historical data.

The project follows the standard 8-step End-to-End Machine Learning workflow.

---

## 2. Problem Statement  

Financial markets are complex and dynamic.  
The goal of this project is to analyze historical S&P 500 data and predict whether the index will increase or decrease on the next trading day.

This is formulated as a binary classification problem:

- 1 → Market goes Up  
- 0 → Market goes Down  

---

# 3. End-to-End Machine Learning Workflow  

## 3.1 Look at the Big Picture  

The main objective is to predict next-day price direction based on historical values of the S&P 500 index.

The problem is treated as a supervised binary classification task.

---

## 3.2 Get the Data  

The dataset (SP500.csv) contains historical S&P 500 index values.

Main columns include:

- observation_date  
- SP500  

The dataset is stored inside the `data/` folder.

---

## 3.3 Explore and Visualize the Data  

Exploratory Data Analysis (EDA) was performed to:

- Understand data distribution  
- Inspect trends over time  
- Check for missing values  
- Examine feature behavior  

Visualization libraries used:

- Matplotlib  
- Seaborn  

---

## 3.4 Prepare the Data  

Data preprocessing steps included:

- Converting the date column to datetime format  
- Sorting values by date  
- Creating a target column using next-day direction  
- Dropping missing values  
- Splitting features (X) and target (y)  
- Performing train-test split  

Target creation logic:

Target = 1 if next day value > current day value  
Target = 0 otherwise  

---

## 3.5 Model Selection and Training  

The following classification models were trained:

- Logistic Regression  
- Decision Tree  
- Random Forest  

Models were trained using the training dataset and evaluated using the test dataset.

---

## 3.6 Model Fine-Tuning  

Hyperparameter tuning was applied using GridSearchCV to improve model performance.

The best model was selected based on evaluation metrics.

---

## 3.7 Model Evaluation  

Models were evaluated using:

- Accuracy Score  
- Confusion Matrix  
- Precision  
- Recall  
- F1-Score  

These metrics help assess prediction quality and classification balance.

---

## 3.8 Deployment and Maintenance (Conceptual)  

In a real-world scenario, the trained model could be deployed using:

- Flask  
- FastAPI  

The deployed system would:

- Continuously collect new market data  
- Generate daily predictions  
- Monitor model performance  
- Retrain when performance declines  

---

# 4. Technologies Used  

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Seaborn  
- Jupyter Notebook  

---

# 5. Project Structure  

```
ABDALLAH-MAKNI_STOCK-PRICE-CLASSIFIER/

│── data/
│     └── SP500.csv
│
│── notebooks/
│     └── stock_model.ipynb
│
│── README.md
│
│── requirements.txt
```

---

# 6. How to Run the Project  

1. Clone the repository  

2. Install required libraries:

```
pip install -r requirements.txt
```

3. Open Jupyter Notebook:

```
jupyter notebook
```

4. Open `stock_model.ipynb` and run all cells sequentially.

---

# 7. Results  

The final selected model achieved strong classification performance in predicting next-day market direction.

The results demonstrate that Machine Learning models can capture useful patterns in historical financial data.

---

# 8. Conclusion  

This project demonstrates the full implementation of an end-to-end Machine Learning workflow for financial market prediction.

It covers:

- Data preprocessing  
- Feature engineering  
- Model comparison  
- Hyperparameter tuning  
- Model evaluation  

The project satisfies the AI-FUN-2026 hands-on AI/ML project requirements.