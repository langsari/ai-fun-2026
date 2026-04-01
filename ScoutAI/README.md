# ⚽ ScoutAI: Football Player Recommendation & Playstyle Matching System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data_Manipulation-green)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626)
![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-20BEFF)


## 1. Project Overview
**ScoutAI** is a data-driven Decision Support System (DSS) designed for football club scouting departments. Inspired by the "Moneyball" philosophy, this system utilizes Machine Learning to identify "undervalued assets"—players whose statistical "Playstyle DNA" closely matches a specific target player, allowing clubs to find perfect tactical replacements efficiently.

## 2. Problem Statement
**Business Problem:** Traditional football scouting heavily relies on the "eye-test," which is prone to human cognitive bias. Expensive transfer flops can severely damage a club's financial stability and tactical setup.

**Technical Challenge:** This project strictly enforces a **"No-Drop Policy,"** requiring the use of all **98 multidimensional attributes** without applying traditional feature selection (e.g., dropping columns). Processing 98 dimensions introduces the **"Curse of Dimensionality"** and significant dimensional noise, causing traditional distance metrics to fail if not properly handled.

## 3. Dataset
* **Source:** Football Manager 2023 Dataset
* **Volume:** 8,452 player profiles.
* **Features:** **98 distinct columns** encompassing:
* **Technical:** Finishing, Passing, Tackling, Crossing, etc.
* **Mental:** Anticipation, Composure, Vision, Work Rate, etc.
* **Physical:** Pace, Stamina, Strength, Agility, etc.
* **Financial & Reputation:** Market Value, Salary, World Reputation.

## 4. Data Exploration / Visualization (EDA)
Initial exploratory data analysis revealed two critical insights:
1. **Scale Imbalance:** On-pitch attributes range from 1 to 20, whereas financial metrics (salary/values) span into the millions. Without scaling, financial data would completely dominate the algorithms.
2. **Conflicting Features:** Goalkeeper (GK) specific attributes (e.g., Handling, Reflexes) act as mathematical noise when searching for outfield players, and vice versa. 

## 5. Model Selection & Training
To overcome the No-Drop Policy and dimensionality issues, the data preparation phase features a core innovation before model training:

### 🌟 The Secret Sauce: Dynamic Feature Weighting
After standardizing all 98 dimensions using `StandardScaler` (Z-score normalization), we applied **Contextual Feature Scaling**:
* **Outfield Targets:** GK attributes are suppressed to a **1% (0.01)** weight.
* **Goalkeeper Targets:** GK attributes are amplified by **200% (2.00)**, while outfield skills are suppressed to 1%.
* **Financial Constraints:** Market value and salary weights are penalized to **20% (0.20)** to prioritize on-pitch playstyle DNA over financial status.

### 🤖 The 5 AI Engines
We deployed a multi-model ensemble to evaluate varying mathematical hypotheses:
1. **K-Nearest Neighbors (K-NN):** Euclidean distance to find exact statistical clones (Overall matching).
2. **Radius NN:** Euclidean distance with a strict radial boundary filter.
3. **Cosine Similarity:** Measures angular distance to identify proportional "Playstyle DNA", independent of overall ability magnitude.
4. **K-Means Clustering:** Categorizes players into tactical archetypes.
5. **DBSCAN:** Density-based clustering to isolate statistical anomalies (Outliers/Generational talents).

## 6. Model Evaluation
Given the unsupervised nature of the problem, we ensured scientific rigor and optimization through the following methods:
* **GridSearchCV via Proxy Task:** For K-NN, we engineered a proxy classification task (predicting the player's general position) with 3-fold cross-validation to find the optimal parameter. Result: `K=7`.
* **Silhouette Analysis:** For K-Means, we iteratively evaluated cluster cohesion and separation, identifying `K=10` as the optimal cluster count.
* **100% Reproducibility:** The pseudo-random number generator is strictly locked (`np.random.seed(42)`) to ensure that all parameter tuning and output results are completely deterministic.

## 7. Results
The system outputs a **Visual Analytics Dashboard** displaying the Top 5 shortlist from each algorithm, converted into intuitive Match Percentages (%).
* **Euclidean Limitations:** K-NN and Radius NN yield lower match rates (40-50%) due to the strictness of 98-dimensional magnitude matching.
* **Cosine Superiority:** **Cosine Similarity** successfully maps proportional playstyle DNA, frequently yielding higher matches (**70-80%+**) and successfully uncovering undervalued gems who play identically to the target but possess lower overall current ability (CA).
* **Outlier Detection:** DBSCAN mathematically validates that certain world-class targets exist in low-density spatial regions, acting as an outlier detector when no 1-to-1 statistical replacement exists.

## 8. How to Run the Project

### Prerequisites
* Python 3.8 or higher
* Jupyter Notebook

### Installation
1.Clone the repository:
   ```bash
   git clone https://github.com/BetterCallYee/ScoutAI.git
   cd ScoutAI
   ```

2.Create & Activate Virtual Environment (Recommended)
   ```bash
   python -m venv venv
   ```
### Activate it:
Windows
   ```bash
   .\venv\Scripts\activate   
   ``` 

Mac / Linux
   ```bash
   source venv/bin/activate
   ```

3.Install Requirements
   ```bash
   pip install -r requirements.txt
   ```
Register the environment as a Jupyter kernel (run once):
   ```bash
   python -m ipykernel install --user --name scoutai --display-name "Python (ScoutAI)"
   ```

### Select Python Kernel
Open ai.ipynb and select:
   ```bash
   Python (ScoutAI)
   ```
from the notebook kernel menu.

### Execution
Run the Jupyter Notebook to explore the full pipeline:
   ```bash
   jupyter notebook ai.ipynb
   ```
Note: The notebook contains an interactive cell where you can input a target player's name (e.g., "Phil Jones", "Alisson") to generate the real-time scouting radar dashboard.

Author: Rifhan Hajiteh & Isarn Ilacharn
Student ID: 661431009 & 661431021
