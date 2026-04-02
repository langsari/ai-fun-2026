This project applies Machine Learning and Deep Learning techniques using TensorFlow/Keras to predict student final grades (G3) and classify their academic risk levels.

The system is built based on a real-world student dataset that includes academic performance, study behavior, family background, and lifestyle factors. Before model development, correlation analysis was performed to understand relationships between features and ensure meaningful feature selection.

Two types of models were implemented and compared:
- Linear Regression (baseline model)
- Deep Learning model using TensorFlow/Keras

The evaluation results show that Linear Regression achieved better performance (R² ≈ 0.78), while the Deep Learning model achieved slightly lower performance (R² ≈ 0.65). This is mainly due to the relatively small dataset size, where simpler models tend to generalize better than more complex models.

The system also includes an interactive web application built with Streamlit, allowing users to input student data and receive real-time predictions. The output includes both the predicted final grade and a risk classification:
- 🔴 High Risk (likely to fail)
- 🟠 Medium Risk
- 🟢 Low Risk (performing well)

To improve usability and performance, the trained model is saved and loaded directly in the application, eliminating the need for retraining during runtime.

## 🚀 How to Run

Install dependencies:pip install -r requirements.txt

Run the application:streamlit run app.py


## 💡 Key Insight

This project demonstrates that model complexity does not always lead to better performance. For smaller datasets, simpler models such as Linear Regression can outperform Deep Learning models.

## 🔮 Future Improvements

Future work may include expanding the dataset, applying advanced feature selection techniques, and tuning hyperparameters using methods such as GridSearch to improve model performance.