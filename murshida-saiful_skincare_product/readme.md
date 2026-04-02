
---

# Skincare Product Recommendation System

## Author

Murshida Saifulislam

---

## Project Overview

This project is an **AI-based skincare recommendation system** that helps users find suitable products based on their:

* Skin Type (oily, dry, sensitive, normal)
* Skin Concern (acne, dark spots, brightening, anti-aging)
* Budget (in Thai Baht - THB)

The system recommends the **Top 10 products** that best match the user's needs while staying within budget.

---

## Problem Statement

Choosing skincare products is difficult due to:

* A large number of available products
* Complex ingredient lists
* Wide variation in prices

This project aims to simplify the decision-making process using AI.

---

## Solution Approach

We implement a **Content-Based Recommendation System** using:

* **TF-IDF (Term Frequency - Inverse Document Frequency)**
  → Converts ingredient text into numerical vectors

* **Cosine Similarity**
  → Measures similarity between user concerns and product ingredients

---

## Project Pipeline

1. **Data Collection**

   * Sephora Dataset (Kaggle)

2. **Data Cleaning**

   * Remove missing values
   * Standardize column names

3. **Feature Engineering**

   * Generate `skin_type` from product highlights

4. **User Input (UI)**

   * Dropdown selection using ipywidgets

5. **Filtering**

   * Budget filtering (THB → USD conversion)
   * Skin type filtering

6. **Modeling**

   * TF-IDF vectorization
   * Cosine similarity ranking

7. **Recommendation Output**

   * Top 10 products

---

## Dataset

* Source: Sephora Dataset
* Sample Size: **100 products**
* Features used:

  * `product_name`
  * `brand_name`
  * `ingredients`
  * `price_usd`
  * `highlights`

---

## Installation

Install required libraries:

```bash
pip install pandas scikit-learn ipywidgets
```

---

## Usage

1. Open the Jupyter Notebook
2. Run all cells
3. Select:

   * Skin Type
   * Skin Concern
   * Budget (THB)
4. Click **"Recommend Products"**

👉 The system will display the Top 10 recommended products.

---

## Example Code

```python
result = recommend(df, 'oily', 'acne', 500)
result
```

---

## Evaluation Metrics

* **Precision@10**
  Measures how many of the top 10 recommendations are relevant

* **Budget Compliance**
  Ensures all recommendations are within the user’s budget

* **Coverage**
  Evaluates diversity of recommended products

---

## Features

* Interactive dropdown UI (no manual typing)
* Budget input in THB
* Real-time recommendations
* Clean and simple interface

---

## Future Improvements

* Improve skin concern matching using advanced NLP (e.g., BERT)
* Build a web application using Streamlit
* Add product images and descriptions
* Personalization based on user history

---

## Conclusion

This project demonstrates how AI can:

* Simplify skincare selection
* Provide personalized recommendations
* Improve user experience

---
