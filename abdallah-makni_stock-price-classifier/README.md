# AI-FUN-2026 Project
Artificial Intelligence Fundamental - Hands-On AI and ML Project with Python

## Student Information
- Name: Abdallah Makni
- Student ID: 671431030
- Course: Artificial Intelligence Fundamental
- Year: 2026
- Project Folder: `abdallah-makni_stock-price-classifier`

## Project Title
Stock Price Movement Classification (S&P 500)

## 1. Environment Setup Instructions
### 1.1. Project Folder
This project is implemented inside:
- `abdallah-makni_stock-price-classifier`

### 1.2. Standardized Structure
Current structure in this project:
```text
abdallah-makni_stock-price-classifier/
|-- data/
|   `-- SP500.csv
|-- notebooks/
|   `-- stock_model1.ipynb
|-- train_improved.py
|-- requirements.txt
`-- README.md
```

### 1.3. Install Environment
```bash
pip install -r requirements.txt
```

## 2. Project Instructions (Applied to This Project)
This project follows the standard 8-step End-to-End ML workflow:

### 2.1. Look at the big picture
Goal: classify market direction using historical S&P 500 data.

### 2.2. Get the data
Data source file:
- `data/SP500.csv`

Main columns:
- `observation_date`
- `SP500`

### 2.3. Explore and visualize the data
EDA and model comparison are provided in:
- `notebooks/stock_model1.ipynb`

### 2.4. Prepare the data
Preparation steps used in code:
- Parse date column and sort by date
- Create target label
- Build technical features (`ret_*`, moving-average ratios, volatility, momentum, day of week)
- Drop missing rows from rolling/shift features
- Use time-based split (80/20)

### 2.5. Select a model and train it
Implemented in:
- `train_improved.py`

Current baseline model:
- Logistic Regression + StandardScaler (Pipeline)

### 2.6. Fine-tune your model
Tuning and iteration were done by testing:
- Feature set changes
- Prediction horizon setting
- Time-series split behavior

### 2.7. Present your solution
Current reported validation result from `train_improved.py`:
- Validation Accuracy: **61%**
- Raw Accuracy: **0.6172**

### 2.8. Launch, monitor, and maintain
Conceptual next step for production:
- Run daily data update
- Re-train/re-validate regularly
- Monitor drift and model performance over time

## 3. Project Submission and Presentation
### 3.1. Repository Submission
Submit code, notebook, and documentation in the AI-FUN-2026 repository under this project folder.

### 3.2. Progress Presentation
Mid-point presentation should include:
- Data understanding
- EDA highlights
- Initial model results

### 3.3. Final Presentation
Final presentation should include:
- Full end-to-end workflow
- Final model metrics and comparison
- Final conclusion and future improvements

## How to Run
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Run improved training:
```bash
python train_improved.py
```
3. Open notebook (optional):
```bash
jupyter notebook
```
Then open `notebooks/stock_model1.ipynb`.
