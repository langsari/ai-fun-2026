# 📱 Mobile Recommendation System

## 🧠 Overview

This project presents a Mobile Recommendation System developed using fundamental Machine Learning techniques. The system is designed to recommend smartphones that best match user preferences based on selected specifications.

The recommendation mechanism is based on **Cosine Similarity**, which measures the similarity between user input and available products in the dataset.

---

## 🎯 Objectives

* To assist users in selecting smartphones that meet their budget and specification requirements
* To apply fundamental concepts of Data Analysis and Machine Learning
* To develop an interactive system using Jupyter Notebook

---

## 📊 Dataset Description

The dataset consists of multiple smartphone attributes, including:

* `price_thb` – Price in Thai Baht
* `ram_gb` – RAM capacity (GB)
* `storage_gb` – Internal storage (GB)
* `screen_size_inch` – Screen size (inches)
* `battery_mah` – Battery capacity (mAh)
* `camera_mp` – Camera resolution (megapixels)
* `support_5g` – 5G support (Yes/No)

---

## ⚙️ Technologies and Tools

The following tools and libraries were used in this project:

* Python
* Pandas (data manipulation and analysis)
* Matplotlib (data visualization)
* Scikit-learn (similarity computation)
* ipywidgets (interactive user interface)

---

## 📈 Data Analysis

Exploratory Data Analysis (EDA) was conducted to better understand the dataset.
The analysis includes:

* Distribution of smartphone prices
* Frequency distribution of RAM configurations
* Relationship between price and camera resolution

These visualizations help provide insights into the dataset and support the recommendation process.

---

## 🎛️ User Interface

An interactive user interface was implemented using ipywidgets, allowing users to specify their preferences, including:

* Budget range
* RAM size
* Storage capacity
* Screen size
* Battery capacity
* Camera quality
* 5G support

---

## 🤖 Recommendation Methodology

The recommendation system follows these steps:

1. Collect user input through the interface
2. Transform input into a structured data format
3. Compute similarity scores using Cosine Similarity
4. Rank smartphones based on similarity scores
5. Return the top 5 most relevant smartphone recommendations

---

## 📌 Example Output

The system outputs a list of recommended smartphones along with their specifications, such as:

* Brand and model
* Price
* RAM and storage
* Camera resolution
* Battery capacity
* 5G support

---

## 🚀 How to Run the Project

1. Open the `.ipynb` file in Jupyter Notebook or Visual Studio Code
2. Run all cells sequentially
3. Adjust the input parameters using the interactive widgets
4. Click the “Recommend” button
5. View the recommended smartphones

---

## ⚠️ Requirements

Ensure that the dataset file `mobile_data.csv` is located in the same directory as the notebook.

Install required libraries using:

```bash
pip install pandas scikit-learn matplotlib ipywidgets
```

---

## 👩‍💻 Author

Nisreen Seeba

---

## 💡 Future Improvements

* Enhance the user interface for better usability
* Incorporate additional filtering options (e.g., brand preference)
* Deploy the system as a web-based application
* Apply advanced recommendation algorithms

---

