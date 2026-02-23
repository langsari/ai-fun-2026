# Cryptocurrency Halal/Haram Classifier

**Author**: Prosperous01  
**Course**: Artificial Intelligence Fundamental  
**Institution**: FT 
**Date**: February 2026

## Project Overview

This machine learning project automatically classifies cryptocurrencies as Halal or Haram based on Islamic finance principles (Shariah compliance). The system uses a Random Forest classifier to analyze key features that determine whether a cryptocurrency is permissible for Muslim investors.

## Problem Statement

Muslim investors need guidance on which cryptocurrencies comply with Islamic finance principles. Manual evaluation of each cryptocurrency is time-consuming and requires deep knowledge of both Islamic finance and blockchain technology. This project aims to automate the classification process.

## Dataset

- **Size**: 30 cryptocurrencies
- **Features**: 7 Shariah compliance indicators
- **Labels**: Binary classification (Halal = 1, Haram = 0)

### Features Used:
1. **has_interest_mechanism** - Presence of Riba (interest)
2. **has_gambling_elements** - Presence of Maysir (gambling)
3. **excessive_speculation** - Level of Gharar (uncertainty)
4. **has_real_utility** - Real-world use case
5. **transparency_score** - Project transparency (1-10 scale)
6. **asset_backed** - Backed by real assets
7. **defi_lending** - Involvement in interest-based lending

### Sample Cryptocurrencies:
- Halal: Bitcoin, Ethereum, Cardano, Solana, Polygon, etc.
- Haram: Dogecoin, Uniswap, Aave, Compound, PancakeSwap

## Methodology

### 1. Data Preparation
- Created labeled dataset based on Islamic finance criteria
- Split data: 70% training, 30% testing

### 2. Model Selection
- Algorithm: Random Forest Classifier
- Reason: Handles non-linear relationships, provides feature importance

### 3. Training
- 100 decision trees (n_estimators=100)
- Random state=42 for reproducibility

### 4. Evaluation Metrics
- Accuracy Score
- Precision and Recall
- Confusion Matrix
- Feature Importance Analysis

## Results

### Model Performance
- **Accuracy**: ~89% (varies based on random split)
- **Precision**: High for both Halal and Haram classes
- **Recall**: Balanced across both classes

### Key Findings
The most important features for classification are:
1. Interest mechanisms (has_interest_mechanism)
2. DeFi lending involvement (defi_lending)
3. Transparency score
4. Real utility

### Example Prediction
A new DeFi lending platform with interest mechanisms:
- **Prediction**: HARAM
- **Confidence**: ~80%

## Project Structure

```
crypto-halal-classifier/
│
├── data/
│   ├── raw/              # Original datasets
│   └── processed/        # Cleaned data
│
├── notebooks/            # Jupyter notebooks for exploration
│
├── src/                  # Source code
│   └── crypto_classifier.py
│
├── models/               # Saved trained models
│
├── reports/              # Results and visualizations
│   └── figures/
│       └── crypto_classifier_results.png
│
├── README.md             # Project documentation
└── requirements.txt      # Python dependencies
```

## Installation & Usage

### Prerequisites
- Python 3.12 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/langsari/ai-fun-2026.git
cd ai-fun-2026/yourname_crypto-halal-classifier
```

2. Create virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Run the Classifier

```bash
python src/crypto_classifier.py
```

### Output
- Console: Detailed classification report
- File: `crypto_classifier_results.png` (visualizations)

## Future Improvements

1. **Expand Dataset**: Collect data on 100+ cryptocurrencies
2. **Real-time Data**: Integrate APIs for live crypto data
3. **Advanced Features**: Add whitepaper NLP analysis
4. **Multiple Models**: Compare SVM, Neural Networks, XGBoost
5. **Expert Validation**: Collaborate with Islamic scholars
6. **Web Interface**: Create user-friendly web app
7. **Continuous Learning**: Update model as new cryptos emerge

## Limitations

- Small dataset (30 samples)
- Simplified feature engineering
- Binary classification (some cryptos may be uncertain)
- No real-time data integration
- Requires expert validation from Islamic scholars

## Islamic Finance Principles

### Key Concepts:
- **Riba (Ribā)**: Interest or usury - strictly prohibited
- **Gharar**: Excessive uncertainty or ambiguity
- **Maysir**: Gambling or speculation
- **Halal**: Permissible under Islamic law
- **Haram**: Forbidden under Islamic law

## Technologies Used

- **Python 3.14**
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning
- **Matplotlib**: Data visualization
- **Seaborn**: Statistical visualization

## References

1. Islamic Finance Principles and Cryptocurrency
2. Shariah Compliance in Digital Assets
3. Random Forest Classification Algorithm
4. CRISP-DM Methodology

## License

This project is for educational purposes as part of the AI Fundamental course.

## Contact

[Prosperous01]

---

**Note**: This classifier is for educational purposes only and should not be used as the sole basis for investment decisions. Always consult with qualified Islamic scholars for religious guidance.