# Deep Learning with Neural Networks
# Add this to: src/deep_learning_classifier.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Deep Learning libraries
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    from tensorflow.keras.utils import to_categorical
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("TensorFlow not installed. Install with: pip install tensorflow")
    TENSORFLOW_AVAILABLE = False

os.makedirs('reports/figures', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("=" * 80)
print("DEEP LEARNING: NEURAL NETWORKS FOR CRYPTO CLASSIFICATION")
print("=" * 80)

if not TENSORFLOW_AVAILABLE:
    print("\n⚠️  TensorFlow is required for deep learning.")
    print("Install it with: pip install tensorflow")
    exit()

# Load data
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
y = df['is_halal'].values

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nDataset: {len(df)} cryptocurrencies")
print(f"Features: {X.shape[1]}")
print(f"Training: {len(X_train)}, Testing: {len(X_test)}")

# Model 1: Simple Deep Neural Network
print("\n" + "=" * 80)
print("MODEL 1: SIMPLE DEEP NEURAL NETWORK")
print("=" * 80)

model_simple = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model_simple.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nModel Architecture:")
model_simple.summary()

# Early stopping
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

print("\nTraining Simple DNN...")
history_simple = model_simple.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=8,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0
)

# Evaluate
y_pred_simple = (model_simple.predict(X_test_scaled) > 0.5).astype(int)
accuracy_simple = accuracy_score(y_test, y_pred_simple)
f1_simple = f1_score(y_test, y_pred_simple)

print(f"\n✓ Training Complete!")
print(f"Test Accuracy: {accuracy_simple*100:.2f}%")
print(f"F1-Score: {f1_simple*100:.2f}%")

# Model 2: Advanced Deep Neural Network
print("\n" + "=" * 80)
print("MODEL 2: ADVANCED DEEP NEURAL NETWORK")
print("=" * 80)

model_advanced = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model_advanced.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nModel Architecture:")
model_advanced.summary()

print("\nTraining Advanced DNN...")
history_advanced = model_advanced.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=8,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0
)

# Evaluate
y_pred_advanced = (model_advanced.predict(X_test_scaled) > 0.5).astype(int)
accuracy_advanced = accuracy_score(y_test, y_pred_advanced)
f1_advanced = f1_score(y_test, y_pred_advanced)

print(f"\n✓ Training Complete!")
print(f"Test Accuracy: {accuracy_advanced*100:.2f}%")
print(f"F1-Score: {f1_advanced*100:.2f}%")

# Comparison with traditional ML
print("\n" + "=" * 80)
print("COMPARISON: TRADITIONAL ML VS DEEP LEARNING")
print("=" * 80)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Train traditional models
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)
gb_acc = accuracy_score(y_test, gb_pred)

comparison = pd.DataFrame({
    'Model': ['Random Forest', 'Gradient Boosting', 'Simple DNN', 'Advanced DNN'],
    'Accuracy': [rf_acc, gb_acc, accuracy_simple, accuracy_advanced],
    'F1-Score': [
        f1_score(y_test, rf_pred),
        f1_score(y_test, gb_pred),
        f1_simple,
        f1_advanced
    ]
})

print(comparison.to_string(index=False))

# Visualizations
fig = plt.figure(figsize=(18, 12))

# 1. Training History - Simple DNN
ax1 = plt.subplot(3, 3, 1)
ax1.plot(history_simple.history['accuracy'], label='Train Accuracy', linewidth=2)
ax1.plot(history_simple.history['val_accuracy'], label='Val Accuracy', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.set_title('Simple DNN - Training History', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Training History - Advanced DNN
ax2 = plt.subplot(3, 3, 2)
ax2.plot(history_advanced.history['accuracy'], label='Train Accuracy', linewidth=2)
ax2.plot(history_advanced.history['val_accuracy'], label='Val Accuracy', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Advanced DNN - Training History', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Loss History - Simple DNN
ax3 = plt.subplot(3, 3, 4)
ax3.plot(history_simple.history['loss'], label='Train Loss', linewidth=2)
ax3.plot(history_simple.history['val_loss'], label='Val Loss', linewidth=2)
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Loss')
ax3.set_title('Simple DNN - Loss History', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Loss History - Advanced DNN
ax4 = plt.subplot(3, 3, 5)
ax4.plot(history_advanced.history['loss'], label='Train Loss', linewidth=2)
ax4.plot(history_advanced.history['val_loss'], label='Val Loss', linewidth=2)
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Loss')
ax4.set_title('Advanced DNN - Loss History', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Model Accuracy Comparison
ax5 = plt.subplot(3, 3, 3)
models = ['RF', 'GB', 'Simple\nDNN', 'Advanced\nDNN']
accuracies = comparison['Accuracy'].tolist()
colors = ['#3498db', '#3498db', '#e74c3c', '#e74c3c']
bars = ax5.bar(models, accuracies, color=colors, alpha=0.7, edgecolor='black')
ax5.set_ylabel('Accuracy')
ax5.set_title('Model Comparison', fontweight='bold')
ax5.set_ylim([0.85, 1.0])
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height + 0.005,
            f'{acc*100:.1f}%', ha='center', va='bottom', fontsize=9)

# 6. Confusion Matrix - Advanced DNN
ax6 = plt.subplot(3, 3, 6)
cm = confusion_matrix(y_test, y_pred_advanced)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Haram', 'Halal'], 
            yticklabels=['Haram', 'Halal'], ax=ax6)
ax6.set_ylabel('Actual')
ax6.set_xlabel('Predicted')
ax6.set_title('Confusion Matrix - Advanced DNN', fontweight='bold')

# 7. Classification Report - Advanced DNN
ax7 = plt.subplot(3, 3, 7)
report = classification_report(y_test, y_pred_advanced, target_names=['Haram', 'Halal'], output_dict=True)
report_df = pd.DataFrame(report).transpose().iloc[:-3, :3]
sns.heatmap(report_df, annot=True, fmt='.2f', cmap='YlGnBu', ax=ax7, cbar=False)
ax7.set_title('Classification Metrics - Advanced DNN', fontweight='bold')

# 8. F1-Score Comparison
ax8 = plt.subplot(3, 3, 8)
f1_scores = comparison['F1-Score'].tolist()
bars = ax8.bar(models, f1_scores, color=colors, alpha=0.7, edgecolor='black')
ax8.set_ylabel('F1-Score')
ax8.set_title('F1-Score Comparison', fontweight='bold')
ax8.set_ylim([0.85, 1.0])
for bar, f1 in zip(bars, f1_scores):
    height = bar.get_height()
    ax8.text(bar.get_x() + bar.get_width()/2., height + 0.005,
            f'{f1*100:.1f}%', ha='center', va='bottom', fontsize=9)

# 9. Model Architecture Comparison
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')
architecture_text = """
SIMPLE DNN:
• Input (7) → Dense(64) → Dropout(0.3)
• Dense(32) → Dropout(0.2)
• Dense(16) → Output(1)
• Total params: ~3,000

ADVANCED DNN:
• Input (7) → Dense(128) → BatchNorm → Dropout(0.4)
• Dense(64) → BatchNorm → Dropout(0.3)
• Dense(32) → BatchNorm → Dropout(0.2)
• Dense(16) → Output(1)
• Total params: ~12,000
"""
ax9.text(0.1, 0.5, architecture_text, fontsize=10, family='monospace',
         verticalalignment='center')
ax9.set_title('Neural Network Architectures', fontweight='bold')

plt.tight_layout()
plt.savefig('reports/figures/deep_learning_results.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved: reports/figures/deep_learning_results.png")

# Save models
model_simple.save('models/simple_dnn_model.h5')
model_advanced.save('models/advanced_dnn_model.h5')
print("\n✓ Models saved:")
print("  - models/simple_dnn_model.h5")
print("  - models/advanced_dnn_model.h5")

print("\n" + "=" * 80)
print("DEEP LEARNING COMPLETE!")
print("=" * 80)
print(f"\n🏆 Best Deep Learning Model: {'Advanced DNN' if accuracy_advanced > accuracy_simple else 'Simple DNN'}")
print(f"   Accuracy: {max(accuracy_simple, accuracy_advanced)*100:.2f}%")
print(f"\n📊 Comparison with Traditional ML:")
print(f"   Random Forest: {rf_acc*100:.2f}%")
print(f"   Gradient Boosting: {gb_acc*100:.2f}%")
print(f"   Deep Learning (Best): {max(accuracy_simple, accuracy_advanced)*100:.2f}%")