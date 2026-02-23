import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Create a sample dataset
data = {
    'crypto_name': [
        'Bitcoin', 'Ethereum', 'Tether', 'BNB', 'XRP',
        'Cardano', 'Dogecoin', 'Solana', 'USDC', 'Polygon',
        'Polkadot', 'Litecoin', 'Avalanche', 'Chainlink', 'Uniswap',
        'Stellar', 'Algorand', 'VeChain', 'Cosmos', 'Hedera',
        'Fantom', 'Aave', 'Compound', 'PancakeSwap', 'Dai',
        'Near', 'Flow', 'Elrond', 'Zilliqa', 'Harmony'
    ],
    'has_interest_mechanism': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    'has_gambling_elements': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'excessive_speculation': [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'has_real_utility': [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    'transparency_score': [8, 9, 9, 8, 7, 9, 5, 8, 10, 8, 8, 7, 8, 9, 7, 9, 9, 7, 8, 8, 7, 8, 8, 7, 9, 8, 8, 8, 7, 7],
    'asset_backed': [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    'defi_lending': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    'is_halal': [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

print("=" * 60)
print("CRYPTOCURRENCY HALAL/HARAM CLASSIFIER")
print("=" * 60)
print(f"\nDataset Overview:")
print(f"Total cryptocurrencies: {len(df)}")
print(f"Halal: {sum(df['is_halal'])}")
print(f"Haram: {len(df) - sum(df['is_halal'])}")
print("\nFirst 5 rows:")
print(df.head())

X = df.drop(['crypto_name', 'is_halal'], axis=1)
y = df['is_halal']

feature_names = X.columns.tolist()
print(f"\nFeatures used: {feature_names}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"\nTraining set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

print("\n" + "=" * 60)
print("TRAINING THE MODEL...")
print("=" * 60)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

print("✓ Model trained successfully!")

y_pred = clf.predict(X_test)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Haram', 'Halal']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nMost important features for classification:")
print(feature_importance)

print("\n" + "=" * 60)
print("PREDICT NEW CRYPTOCURRENCY")
print("=" * 60)

new_crypto = pd.DataFrame({
    'has_interest_mechanism': [1],
    'has_gambling_elements': [0],
    'excessive_speculation': [0],
    'has_real_utility': [1],
    'transparency_score': [8],
    'asset_backed': [0],
    'defi_lending': [1]
})

prediction = clf.predict(new_crypto)
probability = clf.predict_proba(new_crypto)

print("\nNew Crypto Features:")
print("- Has interest mechanism: Yes")
print("- Has gambling elements: No")
print("- Excessive speculation: No")
print("- Has real utility: Yes")
print("- Transparency score: 8/10")
print("- Asset backed: No")
print("- DeFi lending: Yes")

print(f"\nPrediction: {'HALAL' if prediction[0] == 1 else 'HARAM'}")
print(f"Confidence - Haram: {probability[0][0]*100:.2f}%, Halal: {probability[0][1]*100:.2f}%")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance')
plt.title('Feature Importance in Classification')
plt.tight_layout()
plt.subplot(1, 2, 2)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Haram', 'Halal'], 
            yticklabels=['Haram', 'Halal'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.tight_layout()

import os
os.makedirs('reports/figures', exist_ok=True)
plt.savefig('reports/figures/crypto_classifier_results.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved as 'reports/figures/crypto_classifier_results.png'")

print("\n" + "=" * 60)
print("PROJECT COMPLETE!")
print("=" * 60)