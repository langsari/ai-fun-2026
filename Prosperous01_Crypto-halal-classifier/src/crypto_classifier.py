import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                             precision_score, recall_score, f1_score, roc_auc_score)
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create directories if they don't exist
os.makedirs('reports/figures', exist_ok=True)

# Expanded dataset with 100 cryptocurrencies
data = {
    'crypto_name': [
        # Major cryptocurrencies (1-20)
        'Bitcoin', 'Ethereum', 'Tether', 'BNB', 'XRP', 'Cardano', 'Dogecoin', 'Solana', 
        'USDC', 'Polygon', 'Polkadot', 'Litecoin', 'Avalanche', 'Chainlink', 'Uniswap',
        'Stellar', 'Algorand', 'VeChain', 'Cosmos', 'Hedera',
        
        # DeFi & Lending Platforms (21-35)
        'Aave', 'Compound', 'PancakeSwap', 'SushiSwap', 'Curve', 'Maker', 'Yearn',
        'Synthetix', 'Balancer', 'Bancor', 'dYdX', 'Venus', 'JustLend', 'Euler', 'Morpho',
        
        # Stablecoins (36-45)
        'Dai', 'BUSD', 'TrueUSD', 'Pax Dollar', 'USDD', 'Frax', 'TUSD', 'USDP', 'GUSD', 'LUSD',
        
        # Layer 2 & Scaling (46-55)
        'Arbitrum', 'Optimism', 'zkSync', 'StarkNet', 'Loopring', 'Immutable X', 
        'Boba Network', 'Metis', 'Aztec', 'Scroll',
        
        # Smart Contract Platforms (56-65)
        'Tron', 'EOS', 'NEAR', 'Flow', 'Elrond', 'Zilliqa', 'Harmony', 'Fantom', 
        'Tezos', 'Klaytn',
        
        # Gaming & NFT (66-75)
        'Axie Infinity', 'Decentraland', 'The Sandbox', 'Enjin', 'Gala', 'ApeCoin',
        'Immutable', 'WAX', 'Ultra', 'MyNeighborAlice',
        
        # Oracles & Infrastructure (76-85)
        'Band Protocol', 'API3', 'Tellor', 'DIA', 'Pyth Network', 'Nest Protocol',
        'UMA', 'Razor Network', 'Witnet', 'Umbrella Network',
        
        # Privacy Coins (86-90)
        'Monero', 'Zcash', 'Dash', 'Horizen', 'Beam',
        
        # Meme & Speculative (91-100)
        'Shiba Inu', 'Pepe', 'Floki', 'Baby Doge', 'SafeMoon', 'Dogelon Mars',
        'Kishu Inu', 'Akita Inu', 'Hoge Finance', 'Samoyedcoin'
    ],
    
    # Feature 1: Interest mechanism (Riba)
    'has_interest_mechanism': [
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,  # 1-20
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,  # 21-35 (DeFi/Lending)
        0,0,0,0,0,0,0,0,0,0,  # 36-45 (Stablecoins)
        0,0,0,0,0,0,0,0,0,0,  # 46-55 (Layer 2)
        0,0,0,0,0,0,0,0,0,0,  # 56-65 (Smart contracts)
        0,0,0,0,0,0,0,0,0,0,  # 66-75 (Gaming)
        0,0,0,0,0,0,0,0,0,0,  # 76-85 (Oracles)
        0,0,0,0,0,  # 86-90 (Privacy)
        0,0,0,0,1,0,0,0,0,0   # 91-100 (Meme - SafeMoon has staking rewards)
    ],
    
    # Feature 2: Gambling elements (Maysir)
    'has_gambling_elements': [
        0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,  # 1-20 (Doge has gambling aspect)
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,  # 21-35
        0,0,0,0,0,0,0,0,0,0,  # 36-45
        0,0,0,0,0,0,0,0,0,0,  # 46-55
        0,0,0,0,0,0,0,0,0,0,  # 56-65
        1,1,1,0,1,0,0,0,0,1,  # 66-75 (Gaming platforms have gambling)
        0,0,0,0,0,0,0,0,0,0,  # 76-85
        0,0,0,0,0,  # 86-90
        1,1,1,1,1,1,1,1,1,1   # 91-100 (Meme coins = gambling)
    ],
    
    # Feature 3: Excessive speculation (Gharar)
    'excessive_speculation': [
        1,0,0,0,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,  # 1-20
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,  # 21-35
        0,0,0,0,0,0,0,0,0,0,  # 36-45
        0,0,0,0,0,0,0,0,0,0,  # 46-55
        0,0,0,0,0,0,0,0,0,0,  # 56-65
        1,1,1,0,1,1,0,0,0,1,  # 66-75
        0,0,0,0,0,0,0,0,0,0,  # 76-85
        1,1,1,1,1,  # 86-90 (Privacy = speculation)
        1,1,1,1,1,1,1,1,1,1   # 91-100 (All meme = speculation)
    ],
    
    # Feature 4: Real utility
    'has_real_utility': [
        1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,  # 1-20
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,  # 21-35
        1,1,1,1,1,1,1,1,1,1,  # 36-45
        1,1,1,1,1,1,1,1,1,1,  # 46-55
        1,1,1,1,1,1,1,1,1,1,  # 56-65
        1,1,1,1,1,0,1,1,1,1,  # 66-75
        1,1,1,1,1,1,1,1,1,1,  # 76-85
        1,1,1,1,1,  # 86-90
        0,0,0,0,0,0,0,0,0,0   # 91-100 (Meme = no utility)
    ],
    
    # Feature 5: Transparency score (1-10)
    'transparency_score': [
        8,9,9,8,7,9,5,8,10,8,8,7,8,9,7,9,9,7,8,8,  # 1-20
        8,8,7,7,7,9,7,7,7,7,7,6,6,7,7,  # 21-35
        9,9,8,8,7,8,8,8,9,8,  # 36-45
        8,8,7,7,7,7,7,7,6,7,  # 46-55
        7,6,8,8,8,7,7,7,8,7,  # 56-65
        6,6,6,7,6,5,7,6,6,6,  # 66-75
        7,7,6,7,7,6,6,6,6,6,  # 76-85
        5,6,5,5,5,  # 86-90 (Privacy = low transparency)
        4,3,4,4,3,3,3,3,3,3   # 91-100 (Meme = very low)
    ],
    
    # Feature 6: Asset backed
    'asset_backed': [
        0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,  # 1-20
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,  # 21-35
        1,1,1,1,1,1,1,1,1,1,  # 36-45 (Stablecoins = backed)
        0,0,0,0,0,0,0,0,0,0,  # 46-55
        0,0,0,0,0,0,0,0,0,0,  # 56-65
        0,0,0,0,0,0,0,0,0,0,  # 66-75
        0,0,0,0,0,0,0,0,0,0,  # 76-85
        0,0,0,0,0,  # 86-90
        0,0,0,0,0,0,0,0,0,0   # 91-100
    ],
    
    # Feature 7: DeFi lending
    'defi_lending': [
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,  # 1-20
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,  # 21-35 (All DeFi lending)
        0,0,0,0,0,0,0,0,0,0,  # 36-45
        0,0,0,0,0,0,0,0,0,0,  # 46-55
        0,0,0,0,0,0,0,0,0,0,  # 56-65
        0,0,0,0,0,0,0,0,0,0,  # 66-75
        0,0,0,0,0,0,0,0,0,0,  # 76-85
        0,0,0,0,0,  # 86-90
        0,0,0,0,0,0,0,0,0,0   # 91-100
    ],
    
    # Label: 1 = Halal, 0 = Haram
    'is_halal': [
        1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,1,1,  # 1-20
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,  # 21-35 (All DeFi = Haram)
        1,1,1,1,1,1,1,1,1,1,  # 36-45 (Stablecoins = Halal)
        1,1,1,1,1,1,1,1,1,1,  # 46-55 (Layer 2 = Halal)
        1,1,1,1,1,1,1,1,1,1,  # 56-65 (Smart contracts = Halal)
        0,0,0,1,0,0,1,1,1,0,  # 66-75 (Gaming mixed)
        1,1,1,1,1,1,1,1,1,1,  # 76-85 (Oracles = Halal)
        1,1,1,1,1,  # 86-90 (Privacy = Halal)
        0,0,0,0,0,0,0,0,0,0   # 91-100 (Meme = Haram)
    ]
}

df = pd.DataFrame(data)

print("=" * 80)
print("ENHANCED CRYPTOCURRENCY HALAL/HARAM CLASSIFIER")
print("=" * 80)
print(f"\nDataset Overview:")
print(f"Total cryptocurrencies: {len(df)}")
print(f"Halal: {sum(df['is_halal'])} ({sum(df['is_halal'])/len(df)*100:.1f}%)")
print(f"Haram: {len(df) - sum(df['is_halal'])} ({(len(df)-sum(df['is_halal']))/len(df)*100:.1f}%)")

# Prepare features and labels
X = df.drop(['crypto_name', 'is_halal'], axis=1)
y = df['is_halal']

feature_names = X.columns.tolist()
print(f"\nFeatures used: {feature_names}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print(f"\nTraining set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

# Define multiple models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Support Vector Machine': SVC(kernel='rbf', probability=True, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Naive Bayes': GaussianNB(),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}

print("\n" + "=" * 80)
print("TRAINING AND COMPARING MULTIPLE MODELS")
print("=" * 80)

results = []

for name, model in models.items():
    print(f"\n{name}:")
    print("-" * 40)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Calculate error rate
    error_rate = 1 - accuracy
    
    # ROC AUC (if probability available)
    roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
    
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall: {recall*100:.2f}%")
    print(f"F1-Score: {f1*100:.2f}%")
    print(f"Error Rate: {error_rate*100:.2f}%")
    print(f"Cross-Validation Score: {cv_mean*100:.2f}% (+/- {cv_std*100:.2f}%)")
    if roc_auc:
        print(f"ROC AUC Score: {roc_auc:.4f}")
    
    # Store results
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Error Rate': error_rate,
        'CV Score': cv_mean,
        'CV Std': cv_std,
        'ROC AUC': roc_auc if roc_auc else 0
    })

# Create comparison DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Accuracy', ascending=False)

print("\n" + "=" * 80)
print("MODEL COMPARISON SUMMARY")
print("=" * 80)
print(results_df.to_string(index=False))

# Find best model
best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]

print(f"\n🏆 Best Model: {best_model_name}")
print(f"   Accuracy: {results_df.iloc[0]['Accuracy']*100:.2f}%")
print(f"   Error Rate: {results_df.iloc[0]['Error Rate']*100:.2f}%")

# Detailed evaluation of best model
print("\n" + "=" * 80)
print(f"DETAILED EVALUATION - {best_model_name}")
print("=" * 80)

y_pred_best = best_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=['Haram', 'Halal']))

cm = confusion_matrix(y_test, y_pred_best)
print("\nConfusion Matrix:")
print(cm)

# Feature importance (if available)
if hasattr(best_model, 'feature_importances_'):
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE")
    print("=" * 80)
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.to_string(index=False))

# Create comprehensive visualizations
fig = plt.figure(figsize=(18, 12))

# 1. Model Accuracy Comparison
plt.subplot(3, 3, 1)
plt.barh(results_df['Model'], results_df['Accuracy'])
plt.xlabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.xlim([0, 1])
for i, v in enumerate(results_df['Accuracy']):
    plt.text(v + 0.01, i, f'{v*100:.1f}%', va='center')

# 2. Error Rate Comparison
plt.subplot(3, 3, 2)
plt.barh(results_df['Model'], results_df['Error Rate'], color='red', alpha=0.6)
plt.xlabel('Error Rate')
plt.title('Model Error Rate Comparison')
plt.xlim([0, max(results_df['Error Rate']) * 1.2])
for i, v in enumerate(results_df['Error Rate']):
    plt.text(v + 0.005, i, f'{v*100:.1f}%', va='center')

# 3. F1-Score Comparison
plt.subplot(3, 3, 3)
plt.barh(results_df['Model'], results_df['F1-Score'], color='green', alpha=0.6)
plt.xlabel('F1-Score')
plt.title('Model F1-Score Comparison')
plt.xlim([0, 1])
for i, v in enumerate(results_df['F1-Score']):
    plt.text(v + 0.01, i, f'{v*100:.1f}%', va='center')

# 4. Precision vs Recall
plt.subplot(3, 3, 4)
plt.scatter(results_df['Precision'], results_df['Recall'], s=100, alpha=0.6)
for i, txt in enumerate(results_df['Model']):
    plt.annotate(txt, (results_df.iloc[i]['Precision'], results_df.iloc[i]['Recall']), 
                fontsize=8, ha='right')
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.title('Precision vs Recall')
plt.grid(True, alpha=0.3)

# 5. Cross-Validation Scores
plt.subplot(3, 3, 5)
plt.barh(results_df['Model'], results_df['CV Score'], color='purple', alpha=0.6)
plt.xlabel('CV Score')
plt.title('Cross-Validation Scores')
plt.xlim([0, 1])

# 6. Confusion Matrix (Best Model)
plt.subplot(3, 3, 6)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Haram', 'Halal'], 
            yticklabels=['Haram', 'Halal'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title(f'Confusion Matrix - {best_model_name}')

# 7. Feature Importance (if available)
if hasattr(best_model, 'feature_importances_'):
    plt.subplot(3, 3, 7)
    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.xlabel('Importance')
    plt.title('Feature Importance')

# 8. Dataset Distribution
plt.subplot(3, 3, 8)
labels = ['Halal', 'Haram']
sizes = [sum(df['is_halal']), len(df) - sum(df['is_halal'])]
colors = ['#2ecc71', '#e74c3c']
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('Dataset Distribution')

# 9. ROC AUC Comparison
plt.subplot(3, 3, 9)
roc_data = results_df[results_df['ROC AUC'] > 0]
if len(roc_data) > 0:
    plt.barh(roc_data['Model'], roc_data['ROC AUC'], color='orange', alpha=0.6)
    plt.xlabel('ROC AUC Score')
    plt.title('ROC AUC Comparison')
    plt.xlim([0, 1])

plt.tight_layout()
plt.savefig('reports/figures/enhanced_crypto_classifier_results.png', dpi=300, bbox_inches='tight')
print("\n✓ Comprehensive visualization saved as 'reports/figures/enhanced_crypto_classifier_results.png'")

print("\n" + "=" * 80)
print("PROJECT COMPLETE!")
print("=" * 80)
print(f"\nKey Insights:")
print(f"- Analyzed {len(df)} cryptocurrencies")
print(f"- Compared {len(models)} machine learning models")
print(f"- Best performing model: {best_model_name}")
print(f"- Lowest error rate: {results_df.iloc[0]['Error Rate']*100:.2f}%")
print(f"- Dataset balance: {sum(df['is_halal'])}/{len(df)} Halal")