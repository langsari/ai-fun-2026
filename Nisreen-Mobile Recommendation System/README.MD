# 📱 Mobile Recommendation System

## 🧠 Overview

This project presents a **Mobile Recommendation System** developed using fundamental Machine Learning techniques. The system is designed to recommend smartphones that best match user preferences based on selected specifications.

The recommendation mechanism is based on **Cosine Similarity**, which measures the similarity between the user's input vector and each smartphone in the dataset.

---

## 🎯 Objectives

- To assist users in selecting smartphones that meet their budget and specification requirements
- To apply fundamental concepts of Data Analysis and Machine Learning
- To develop an interactive system using Jupyter Notebook and ipywidgets

---

## 📊 Dataset Description

**File:** `mobile_data.csv`

The dataset consists of **50 smartphone models** from **10 brands**, including the following attributes:

| Column | Description |
|--------|-------------|
| `brand` | Smartphone brand (e.g. Samsung, iPhone, Xiaomi) |
| `model` | Model name |
| `price_thb` | Price in Thai Baht |
| `ram_gb` | RAM capacity (GB) |
| `storage_gb` | Internal storage (GB) |
| `screen_size_inch` | Screen size (inches) |
| `battery_mah` | Battery capacity (mAh) |
| `camera_mp` | Camera resolution (megapixels) |
| `support_5g` | 5G support — encoded as Yes=1, No=0 |

---

## ⚙️ Technologies and Tools

| Tool | Purpose |
|------|---------|
| Python | Main programming language |
| Pandas | Data loading and manipulation |
| Matplotlib | Data visualization |
| Scikit-learn | Cosine Similarity computation |
| ipywidgets | Interactive user interface (sliders, dropdowns, button) |
| Jupyter Notebook / VS Code | Development environment |

Install all dependencies with:

```bash
pip install pandas scikit-learn matplotlib ipywidgets
```

---

## 📈 Data Analysis (EDA)

Three visualizations are included to better understand the dataset:

1. **Price Distribution** — Histogram showing how smartphone prices are spread across the dataset
2. **RAM Distribution** — Bar chart showing how many phones have each RAM configuration
3. **Price vs Camera** — Scatter plot showing the relationship between price and camera resolution

Key findings:
- Most smartphones are in the **5,000–15,000 THB** range
- **RAM 8GB** is the most common configuration (26 models)
- Camera resolution does not strongly correlate with price

---

## 🎛️ User Interface

An interactive interface is built using **ipywidgets**, allowing users to specify:

| Widget | Type | Range / Options |
|--------|------|----------------|
| Budget | IntSlider | 3,000 – 40,000 THB |
| RAM | Dropdown | From dataset values |
| Storage | Dropdown | From dataset values |
| Screen Size | FloatSlider | 4.5 – 7.5 inches |
| Battery | IntSlider | 2,000 – 7,000 mAh |
| Camera | IntSlider | 10 – 120 MP |
| 5G Support | ToggleButtons | Yes / No |

After setting preferences, users click **"🔍 Recommend mobile models"** to get results.

---

## 🤖 Recommendation Methodology

The system follows these steps:

1. Load `mobile_data.csv` using Pandas
2. Encode `support_5g` column: `Yes → 1`, `No → 0`
3. Select 7 numeric features for similarity computation
4. User sets preferences via interactive widgets
5. User input is converted into a single-row DataFrame
6. **Cosine Similarity** is computed between user input and all phones in the dataset
7. Smartphones are ranked by similarity score (highest first)
8. **Top 5 most similar smartphones** are displayed with full specs

### Features used for similarity:
```
price_thb, ram_gb, storage_gb, screen_size_inch,
battery_mah, camera_mp, support_5g
```

---

## 📌 Example Output

```
📱 รุ่นมือถือที่แนะนำ:

Samsung Galaxy A54
💰 Price: 12990 Bath
📊 RAM: 8GB | Storage: 256GB
📷 Camera: 50MP | 🔋 5000mAh
📶 5G: Yes
--------------------
Xiaomi Poco X5
💰 Price: 9990 Bath
📊 RAM: 6GB | Storage: 128GB
📷 Camera: 48MP | 🔋 5000mAh
📶 5G: Yes
--------------------
...
```

---

## 🚀 How to Run the Project

1. Make sure `mobile_data.csv` is in the **same folder** as the notebook
2. Open `nisreen_mobile-2.ipynb` in **Jupyter Notebook** or **VS Code**
3. Run all cells from top to bottom (**Run All**)
4. Adjust the sliders and dropdowns to set your preferences
5. Click **"🔍 Recommend mobile models"**
6. View the Top 5 recommended smartphones

---

## ⚠️ Requirements

- Python 3.12+
- `mobile_data.csv` must be in the same directory as the notebook

```bash
pip install pandas scikit-learn matplotlib ipywidgets
```

---

## 💡 Future Improvements

- Add brand preference filtering
- Include more smartphone models and recent releases
- Deploy as a web-based application (e.g. Streamlit)
- Apply more advanced recommendation algorithms (e.g. collaborative filtering)

---

## 👩‍💻 Author

**Nisreen Seeba**  
AI-FUN-2026 — Artificial Intelligence Fundamental  
Hands-On AI and ML Projects with Python
