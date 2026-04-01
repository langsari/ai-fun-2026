# Hyperparameter Tuning with Grid Search
# Add this to: src/hyperparameter_tuning.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, make_scorer, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

os.makedirs('reports/figures', exist_ok=True)

print("=" * 80)
print("HYPERPARAMETER TUNING WITH GRID SEARCH")
print("=" * 80)

# Load data (same as before)
data = {
    'crypto_name': ['Bitcoin', 'Ethereum', 'Tether', 'BNB', 'XRP', 'Cardano', 'Dogecoin', 'Solana', 'USDC', 'Polygon', 'Polkadot', 'Litecoin', 'Avalanche', 'Chainlink', 'Uniswap', 'Stellar', 'Algorand', 'VeChain', 'Cosmos', 'Hedera', 'Aave', 'Compound', 'PancakeSwap', 'SushiSwap', 'Curve', 'Maker', 'Yearn', 'Synthetix', 'Balancer', 'Bancor', 'dYdX', 'Venus', 'JustLend', 'Euler', 'Morpho', 'Dai', 'BUSD', 'TrueUSD', 'Pax Dollar', 'USDD', 'Frax', 'TUSD', 'USDP', 'GUSD', 'LUSD', 'Arbitrum', 'Optimism', 'zkSync', 'StarkNet', 'Loopring', 'Immutable X', 'Boba Network', 'Metis', 'Aztec', 'Scroll', 'Tron', 'EOS', 'NEAR', 'Flow', 'Elrond', 'Zilliqa', 'Harmony', 'Fantom', 'Tezos', 'Klaytn', 'Axie Infinity', 'Decentraland', 'The Sandbox', 'Enjin', 'Gala', 'ApeCoin', 'Immutable', 'WAX', 'Ultra', 'MyNeighborAlice', 'Band Protocol', 'API3', 'Tellor', 'DIA', 'Pyth Network', 'Nest Protocol', 'UMA', 'Razor Network', 'Witnet', 'Umbrella Network', 'Monero', 'Zcash', 'Dash', 'Horizen', 'Beam', 'Shiba Inu', 'Pepe', 'Floki', 'Baby Doge', 'SafeMoon', 'Dogelon Mars', 'Kishu Inu', 'Akita Inu', 'Hoge Finance', 'Samoyedcoin'],
    'has_interest_mechanism': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
    'has_gambling_elements': [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1],
    'excessive_speculation': [1,0,0,0,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    'has_real_utility': [1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0],
    'transparency_score': [8,9,9,8,7,9,5,8,10,8,8,7,8,9,7,9,9,7,8,8,8,8,7,7,7,9,7,7,7,7,7,6,6,7,7,9,9,8,8,7,8,8,8,9,8,8,8,7,7,7,7,7,7,6,7,7,6,8,8,8,7,7,7,8,7,6,6,6,7,6,5,7,6,6,6,7,7,6,7,7,6,6,6,6,6,5,6,5,5,5,4,3,4,4,3,3,3,3,3,3],
    'asset_backed': [0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    'defi_lending': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    'is_halal': [1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,0,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0]
}

df = pd.DataFrame(data)
X = df.drop(['crypto_name', 'is_halal'], axis=1)
y = df['is_halal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

print(f"\nDataset: {len(df)} cryptocurrencies")
print(f"Training: {len(X_train)}, Testing: {len(X_test)}")

# Define hyperparameter grids for top models
print("\n" + "=" * 80)
print("DEFINING HYPERPARAMETER GRIDS")
print("=" * 80)

# Random Forest Grid
rf_param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}

# Gradient Boosting Grid
gb_param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 9],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.8, 0.9, 1.0]
}

print("\nRandom Forest Parameter Grid:")
print(f"Total combinations: {np.prod([len(v) for v in rf_param_grid.values()])}")
for param, values in rf_param_grid.items():
    print(f"  {param}: {values}")

print("\nGradient Boosting Parameter Grid:")
print(f"Total combinations: {np.prod([len(v) for v in gb_param_grid.values()])}")
for param, values in gb_param_grid.items():
    print(f"  {param}: {values}")

# Grid Search for Random Forest
print("\n" + "=" * 80)
print("GRID SEARCH: RANDOM FOREST")
print("=" * 80)

rf_grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=rf_param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

print("\nStarting Grid Search for Random Forest...")
start_time = time.time()
rf_grid_search.fit(X_train, y_train)
rf_time = time.time() - start_time

print(f"\n✓ Grid Search Complete! Time taken: {rf_time:.2f} seconds")
print(f"\nBest Parameters: {rf_grid_search.best_params_}")
print(f"Best CV Score: {rf_grid_search.best_score_*100:.2f}%")

# Evaluate on test set
rf_best = rf_grid_search.best_estimator_
rf_pred = rf_best.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

print(f"\nTest Set Performance:")
print(f"Accuracy: {rf_accuracy*100:.2f}%")
print(f"F1-Score: {rf_f1*100:.2f}%")

# Grid Search for Gradient Boosting
print("\n" + "=" * 80)
print("GRID SEARCH: GRADIENT BOOSTING")
print("=" * 80)

gb_grid_search = GridSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_grid=gb_param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

print("\nStarting Grid Search for Gradient Boosting...")
start_time = time.time()
gb_grid_search.fit(X_train, y_train)
gb_time = time.time() - start_time

print(f"\n✓ Grid Search Complete! Time taken: {gb_time:.2f} seconds")
print(f"\nBest Parameters: {gb_grid_search.best_params_}")
print(f"Best CV Score: {gb_grid_search.best_score_*100:.2f}%")

# Evaluate on test set
gb_best = gb_grid_search.best_estimator_
gb_pred = gb_best.predict(X_test)
gb_accuracy = accuracy_score(y_test, gb_pred)
gb_f1 = f1_score(y_test, gb_pred)

print(f"\nTest Set Performance:")
print(f"Accuracy: {gb_accuracy*100:.2f}%")
print(f"F1-Score: {gb_f1*100:.2f}%")

# Comparison: Default vs Tuned
print("\n" + "=" * 80)
print("COMPARISON: DEFAULT VS HYPERPARAMETER TUNED")
print("=" * 80)

# Train default models for comparison
rf_default = RandomForestClassifier(random_state=42)
rf_default.fit(X_train, y_train)
rf_default_pred = rf_default.predict(X_test)
rf_default_acc = accuracy_score(y_test, rf_default_pred)

gb_default = GradientBoostingClassifier(random_state=42)
gb_default.fit(X_train, y_train)
gb_default_pred = gb_default.predict(X_test)
gb_default_acc = accuracy_score(y_test, gb_default_pred)

comparison = pd.DataFrame({
    'Model': ['Random Forest (Default)', 'Random Forest (Tuned)', 
              'Gradient Boosting (Default)', 'Gradient Boosting (Tuned)'],
    'Accuracy': [rf_default_acc, rf_accuracy, gb_default_acc, gb_accuracy],
    'Improvement': [0, rf_accuracy - rf_default_acc, 0, gb_accuracy - gb_default_acc]
})

print(comparison)

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Accuracy Comparison
ax1 = axes[0, 0]
models = ['RF\nDefault', 'RF\nTuned', 'GB\nDefault', 'GB\nTuned']
accuracies = [rf_default_acc, rf_accuracy, gb_default_acc, gb_accuracy]
colors = ['#3498db', '#2ecc71', '#3498db', '#2ecc71']
bars = ax1.bar(models, accuracies, color=colors, alpha=0.7, edgecolor='black')
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_title('Default vs Tuned Model Accuracy', fontsize=14, fontweight='bold')
ax1.set_ylim([0.85, 1.0])
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{acc*100:.2f}%', ha='center', va='bottom', fontsize=10)

# 2. Improvement Visualization
ax2 = axes[0, 1]
improvements = [(rf_accuracy - rf_default_acc)*100, (gb_accuracy - gb_default_acc)*100]
models_short = ['Random\nForest', 'Gradient\nBoosting']
bars = ax2.bar(models_short, improvements, color=['#2ecc71', '#2ecc71'], alpha=0.7, edgecolor='black')
ax2.set_ylabel('Accuracy Improvement (%)', fontsize=12)
ax2.set_title('Improvement from Hyperparameter Tuning', fontsize=14, fontweight='bold')
ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
for bar, imp in zip(bars, improvements):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
            f'+{imp:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 3. Feature Importance (Tuned Random Forest)
ax3 = axes[1, 0]
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_best.feature_importances_
}).sort_values('importance', ascending=True)
ax3.barh(feature_importance['feature'], feature_importance['importance'], color='orange', alpha=0.7)
ax3.set_xlabel('Importance', fontsize=12)
ax3.set_title('Feature Importance (Tuned Random Forest)', fontsize=14, fontweight='bold')

# 4. Training Time Comparison
ax4 = axes[1, 1]
times = [rf_time, gb_time]
bars = ax4.bar(['Random Forest', 'Gradient Boosting'], times, color=['#e74c3c', '#9b59b6'], alpha=0.7, edgecolor='black')
ax4.set_ylabel('Time (seconds)', fontsize=12)
ax4.set_title('Grid Search Training Time', fontsize=14, fontweight='bold')
for bar, t in zip(bars, times):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{t:.1f}s', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('reports/figures/hyperparameter_tuning_results.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved: reports/figures/hyperparameter_tuning_results.png")

# Save best models
import joblib
os.makedirs('models', exist_ok=True)
joblib.dump(rf_best, 'models/random_forest_tuned.pkl')
joblib.dump(gb_best, 'models/gradient_boosting_tuned.pkl')
print("\n✓ Best models saved:")
print("  - models/random_forest_tuned.pkl")
print("  - models/gradient_boosting_tuned.pkl")

print("\n" + "=" * 80)
print("HYPERPARAMETER TUNING COMPLETE!")
print("=" * 80)
print(f"\n🏆 Best Model: {'Random Forest' if rf_accuracy > gb_accuracy else 'Gradient Boosting'}")
print(f"   Accuracy: {max(rf_accuracy, gb_accuracy)*100:.2f}%")
print(f"   Improvement: +{max(improvements):.2f}%")