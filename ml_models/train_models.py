import pandas as pd
import numpy as np
import glob
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score, accuracy_score)
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

# ── CONFIGURATION ──
PROCESSED_DIR = '../data/processed/'
RESULTS_DIR   = '../results/'
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── LOAD ALL PROCESSED FILES ──
print("Loading processed data...")
all_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, '*.csv')))
print(f"Found {len(all_files)} files\n")

dfs = []
for f in all_files:
    df = pd.read_csv(f)
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
print(f"Total rows loaded: {len(data)}")
print(f"Label distribution:\n{data['label'].value_counts()}\n")

# ── FEATURE SELECTION ──
features = [
    'residual_lat',
    'residual_lon',
    'residual_alt',
    'residual_mag',
    'residual_mag_mean',
    'residual_mag_std',
    'gps_fix_type',
    'gps_satellites',
    'imu_acc_mag',
    'baro_pressure',
]

X = data[features].fillna(0)
y = data['label']

# Get unique labels present in data
unique_labels = sorted(y.unique())
label_names   = {0: 'Normal', 1: 'Spoofing', 2: 'Jamming'}
target_names  = [label_names[l] for l in unique_labels]
n_classes     = len(unique_labels)

print(f"Classes present: {[label_names[l] for l in unique_labels]}")
print(f"Features used: {features}\n")

# ── TRAIN / TEST SPLIT ──
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}\n")

# ── SCALE FEATURES ──
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── RANDOM FOREST ──
print("=" * 50)
print("RANDOM FOREST CLASSIFIER")
print("=" * 50)

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)

rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_f1       = f1_score(y_test, y_pred_rf, average='weighted')

print(f"Accuracy : {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)")
print(f"F1 Score : {rf_f1:.4f}")
print()
print("Classification Report:")
print(classification_report(y_test, y_pred_rf,
      labels=unique_labels, target_names=target_names))

# Confusion matrix
cm_rf = confusion_matrix(y_test, y_pred_rf, labels=unique_labels)
print("Confusion Matrix:")
print(cm_rf)

# Feature importance
print("\nFeature Importances:")
importances = rf.feature_importances_
for feat, imp in sorted(zip(features, importances), key=lambda x: -x[1]):
    print(f"  {feat:<25} {imp:.4f}")

# Cross validation
print("\nCross-validation (5-fold):")
cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=5, scoring='f1_weighted')
print(f"  F1 scores: {cv_scores.round(4)}")
print(f"  Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ── SAVE MODEL ──
with open(os.path.join(RESULTS_DIR, 'random_forest_model.pkl'), 'wb') as f:
    pickle.dump(rf, f)
with open(os.path.join(RESULTS_DIR, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
print("\nRandom Forest model saved.")

# ── PLOT CONFUSION MATRIX ──
fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(cm_rf, cmap='Blues')
ax.set_xticks(range(n_classes))
ax.set_yticks(range(n_classes))
ax.set_xticklabels(target_names)
ax.set_yticklabels(target_names)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title(f'Random Forest Confusion Matrix\nAccuracy: {rf_accuracy*100:.2f}% | F1: {rf_f1:.4f}')
for i in range(n_classes):
    for j in range(n_classes):
        ax.text(j, i, cm_rf[i, j], ha='center', va='center',
                color='white' if cm_rf[i, j] > cm_rf.max()/2 else 'black',
                fontsize=14, fontweight='bold')
plt.colorbar(im)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'rf_confusion_matrix.png'), dpi=150)
plt.close()
print("Confusion matrix plot saved.")

# ── PLOT FEATURE IMPORTANCE ──
fig, ax = plt.subplots(figsize=(8, 5))
sorted_idx = np.argsort(importances)
ax.barh([features[i] for i in sorted_idx], importances[sorted_idx], color='steelblue')
ax.set_xlabel('Importance')
ax.set_title('Random Forest — Feature Importances')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'rf_feature_importance.png'), dpi=150)
plt.close()
print("Feature importance plot saved.")

print("\n" + "=" * 50)
print("RESULTS SUMMARY")
print("=" * 50)
print(f"Random Forest Accuracy : {rf_accuracy*100:.2f}%")
print(f"Random Forest F1 Score : {rf_f1:.4f}")
print(f"\nAll results saved to: {RESULTS_DIR}")
