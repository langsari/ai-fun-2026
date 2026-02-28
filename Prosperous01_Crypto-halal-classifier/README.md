# Cryptocurrency Halal/Haram Classifier

**Author**: Prosperous01  
**Course**: Artificial Intelligence Fundamental  
**Institution**: FT  
**Date**: February 2026

## Project Overview

This machine learning project automatically classifies cryptocurrencies as Halal or Haram based on Islamic finance principles (Shariah compliance). The system compares multiple machine learning algorithms to determine the best model for predicting whether a cryptocurrency is permissible for Muslim investors.

## Problem Statement

Muslim investors need guidance on which cryptocurrencies comply with Islamic finance principles. Manual evaluation of each cryptocurrency is time-consuming and requires deep knowledge of both Islamic finance and blockchain technology. This project aims to automate the classification process using advanced machine learning techniques.

## Dataset

- **Size**: 100 cryptocurrencies (significantly expanded from initial 30)
- **Features**: 7 Shariah compliance indicators
- **Labels**: Binary classification (Halal = 1, Haram = 0)
- **Distribution**: ~62 Halal, ~38 Haram

### Cryptocurrency Categories Included:
1. **Major Cryptocurrencies** - Bitcoin, Ethereum, BNB, Cardano, Solana, etc.
2. **DeFi & Lending Platforms** - Aave, Compound, PancakeSwap, Curve, Maker, etc.
3. **Stablecoins** - USDC, Dai, BUSD, TrueUSD, Frax, etc.
4. **Layer 2 Solutions** - Arbitrum, Optimism, zkSync, StarkNet, Loopring, etc.
5. **Smart Contract Platforms** - Tron, EOS, NEAR, Flow, Elrond, Zilliqa, etc.
6. **Gaming & NFT** - Axie Infinity, Decentraland, Sandbox, Enjin, Gala, etc.
7. **Oracles & Infrastructure** - Chainlink, Band Protocol, API3, Pyth Network, etc.
8. **Privacy Coins** - Monero, Zcash, Dash, Horizen, Beam
9. **Meme Coins** - Shiba Inu, Pepe, Floki, Baby Doge, SafeMoon, etc.

### Features Used:
1. **has_interest_mechanism** - Presence of Riba (interest)
2. **has_gambling_elements** - Presence of Maysir (gambling)
3. **excessive_speculation** - Level of Gharar (uncertainty)
4. **has_real_utility** - Real-world use case
5. **transparency_score** - Project transparency (1-10 scale)
6. **asset_backed** - Backed by real assets
7. **defi_lending** - Involvement in interest-based lending

## Methodology

### 1. Data Preparation
- Created comprehensive labeled dataset based on Islamic finance criteria
- Split data: 75% training, 25% testing
- Ensured balanced representation across categories

### 2. Model Selection & Comparison
Implemented and compared **7 different machine learning algorithms**:

1. **Random Forest** - Ensemble of decision trees
2. **Gradient Boosting** - Sequential ensemble method
3. **Support Vector Machine (SVM)** - Kernel-based classifier
4. **K-Nearest Neighbors** - Instance-based learning
5. **Decision Tree** - Single tree classifier
6. **Naive Bayes** - Probabilistic classifier
7. **Logistic Regression** - Linear classification

### 3. Training & Validation
- 5-Fold Cross-Validation for each model
- Hyperparameters optimized for best performance
- Random state=42 for reproducibility

### 4. Comprehensive Evaluation Metrics
- **Accuracy Score** - Overall correctness
- **Precision** - Correctness of positive predictions
- **Recall** - Coverage of actual positives
- **F1-Score** - Harmonic mean of precision and recall
- **Error Rate** - Percentage of incorrect predictions
- **Cross-Validation Score** - Average accuracy across folds
- **ROC AUC Score** - Quality of probabilistic predictions
- **Confusion Matrix** - Detailed prediction breakdown

## Results

### Model Performance Comparison

The system automatically identifies the best-performing model based on accuracy. Typical results show:

- **Best Model**: Random Forest or Gradient Boosting (typically 92-96% accuracy)
- **Lowest Error Rate**: 4-8%
- **Highest F1-Score**: 0.93-0.97

### Key Findings

**Most Important Features for Classification:**
1. Interest mechanisms (has_interest_mechanism) - ~40% importance
2. DeFi lending involvement (defi_lending) - ~32% importance
3. Transparency score - ~10% importance
4. Gambling elements - ~8% importance

**Classification Patterns:**
- DeFi lending platforms: Consistently classified as HARAM (95%+ confidence)
- Stablecoins: Generally HALAL if asset-backed
- Meme coins: Consistently HARAM due to gambling/speculation
- Privacy coins: Generally HALAL despite low transparency
- Gaming platforms: Mixed results depending on gambling elements

### Detailed Results

All models provide:
- Complete classification reports
- Confusion matrices
- Feature importance rankings (where applicable)
- Cross-validation statistics

## Visualizations

The project generates comprehensive visualizations:

1. **Model Accuracy Comparison** - Bar chart comparing all models
2. **Error Rate Comparison** - Visual comparison of prediction errors
3. **F1-Score Comparison** - Model performance on balanced metric
4. **Precision vs Recall** - Scatter plot showing trade-offs
5. **Cross-Validation Scores** - Reliability across data splits
6. **Confusion Matrix** - Detailed breakdown for best model
7. **Feature Importance** - Which factors matter most
8. **Dataset Distribution** - Pie chart of Halal/Haram split
9. **ROC AUC Comparison** - Quality of probabilistic predictions

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
│       └── enhanced_crypto_classifier_results.png
│
├── README.md             # Project documentation
├── requirements.txt      # Python dependencies
└── .gitignore           # Git ignore file
```

## Installation & Usage

### Prerequisites
- Python 3.12 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/langsari/ai-fun-2026.git
cd ai-fun-2026/prosperous01_crypto-halal-classifier
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
- **Console**: 
  - Dataset statistics
  - Training progress for all 7 models
  - Comprehensive comparison table
  - Best model identification
  - Detailed metrics and reports
  
- **File**: `enhanced_crypto_classifier_results.png` with 9 visualization charts

## Future Improvements

1. **Real-time Data Integration**: Connect to cryptocurrency APIs for live data
2. **Advanced NLP Analysis**: Analyze whitepapers and project documentation
3. **Deep Learning Models**: Implement neural networks for pattern recognition
4. **Ensemble Methods**: Combine multiple models for better predictions
5. **Expert Validation**: Collaborate with Islamic scholars for verification
6. **Web Interface**: Create user-friendly web application
7. **Mobile App**: Develop smartphone app for on-the-go checking
8. **Continuous Learning**: Update model as new cryptos emerge and rules evolve
9. **Multi-class Classification**: Add "Uncertain" category for borderline cases
10. **Explainable AI**: Provide detailed reasoning for each classification

## Limitations

- Dataset based on simplified Shariah criteria
- Binary classification (no "uncertain" category)
- No real-time data integration yet
- Requires expert validation from qualified Islamic scholars
- Some edge cases may need manual review
- Does not account for changing crypto protocols
- Simplified feature engineering (real-world analysis more complex)

## Islamic Finance Principles

### Key Concepts:
- **Riba (Ribā)**: Interest or usury - strictly prohibited in Islam
- **Gharar**: Excessive uncertainty, ambiguity, or risk
- **Maysir**: Gambling, games of chance, or speculation
- **Halal**: Permissible under Islamic law
- **Haram**: Forbidden under Islamic law

### Application to Cryptocurrencies:
- **Interest-based staking/lending** → Haram (Riba)
- **Gambling mechanics in tokens** → Haram (Maysir)
- **Pure speculation with no utility** → Haram (Gharar)
- **Transparent, utility-driven projects** → Potentially Halal
- **Asset-backed stablecoins** → Generally Halal

## Technologies Used

- **Python 3.14** - Programming language
- **Pandas 3.0.1** - Data manipulation and analysis
- **NumPy 2.4.2** - Numerical computing
- **Scikit-learn 1.8.0** - Machine learning algorithms
- **Matplotlib 3.10.8** - Data visualization
- **Seaborn 0.13.2** - Statistical visualization

## Machine Learning Algorithms Explained

1. **Random Forest**: Creates multiple decision trees and combines their predictions
2. **Gradient Boosting**: Builds trees sequentially, each correcting previous errors
3. **SVM**: Finds optimal boundary between classes in high-dimensional space
4. **K-Nearest Neighbors**: Classifies based on similarity to nearby examples
5. **Decision Tree**: Creates a flowchart-like structure for decisions
6. **Naive Bayes**: Uses probability theory based on feature independence
7. **Logistic Regression**: Linear model for binary classification

## References

1. Islamic Finance Principles and Cryptocurrency - Academic Research
2. Shariah Compliance in Digital Assets - AAOIFI Standards
3. Scikit-learn Documentation - Machine Learning Library
4. CRISP-DM Methodology - Data Mining Process
5. Cryptocurrency Market Analysis - Industry Reports

## Academic Context

This project follows the standard machine learning workflow:
1. ✅ Problem Definition - Identify need for automated Shariah compliance
2. ✅ Data Collection - Gather crypto features and labels
3. ✅ Data Exploration - Analyze distribution and patterns
4. ✅ Data Preparation - Clean and structure data
5. ✅ Model Selection - Compare multiple algorithms
6. ✅ Model Training - Train and validate models
7. ✅ Model Evaluation - Comprehensive metrics and comparison
8. ✅ Results Presentation - Clear visualizations and reports

## License

This project is for educational purposes as part of the AI Fundamental course at FT.

## Contact

**Prosperous01**  
GitHub: prosperous01

---

## Acknowledgments

- Course Instructor for guidance and feedback
- Islamic Finance scholars for Shariah principles
- Scikit-learn community for ML tools

---

**Important Disclaimer**: This classifier is for educational and research purposes only. It should not be used as the sole basis for investment decisions. Always consult with qualified Islamic scholars (Ulama) and financial advisors before making investment decisions. The classifications are based on simplified criteria and may not reflect the complete complexity of Islamic jurisprudence (Fiqh).

## Version History

- **v2.0** (Current) - Expanded to 100 cryptos, 7 ML models, comprehensive metrics
- **v1.0** - Initial release with 30 cryptos, single Random Forest model