# fix_labels_and_retrain_lstm.py
# Run from: ~/uav_security_project/
# Purpose: Relabel pre-attack rows in spoofing files as normal (label=0),
#          build temporally-ordered sequences per file, then retrain the LSTM.
# Constraints: Python 3.14, PyTorch only — no TensorFlow, no DroneKit.

import os
import glob
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# ── CONFIG ────────────────────────────────────────────────────────────────────
PROCESSED_DIR    = "data/processed/"
RESULTS_DIR      = "results/"
ATTACK_START_S   = 30.0        # attack begins at t=30s in every spoofing file
SAMPLE_RATE_HZ   = 10          # 10 Hz → 10 rows per second
ATTACK_START_ROW = int(ATTACK_START_S * SAMPLE_RATE_HZ)   # = row 300

WINDOW_SIZE  = 20
HIDDEN_SIZE  = 128
NUM_LAYERS   = 2
DROPOUT      = 0.3
BATCH_SIZE   = 64
EPOCHS       = 40
LR           = 1e-3
NUM_CLASSES  = 3

FEATURES = [
    "residual_lat", "residual_lon", "residual_alt",
    "residual_mag", "residual_mag_mean", "residual_mag_std",
    "gps_fix_type", "gps_satellites", "imu_acc_mag",
    "baro_pressure", "gps_drift_mag", "gps_lat_drift",
    "gps_lon_drift", "gps_imu_lat_diff", "gps_imu_lon_diff"
]

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── STEP 1: RELABEL PRE-ATTACK ROWS ──────────────────────────────────────────
print("=" * 60)
print("STEP 1: Relabelling pre-attack rows in spoofing files")
print("=" * 60)

all_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "*.csv")))
if not all_files:
    raise FileNotFoundError(f"No CSV files found in {PROCESSED_DIR}. "
                            f"Check your path and run from ~/uav_security_project/")

for fpath in all_files:
    fname = os.path.basename(fpath)
    df = pd.read_csv(fpath)

    if "spoofing" in fname and "label" in df.columns:
        before = (df["label"] == 1).sum()
        df.loc[df.index < ATTACK_START_ROW, "label"] = 0
        after  = (df["label"] == 1).sum()
        df.to_csv(fpath, index=False)
        print(f"  {fname}: spoofing rows {before} -> {after}  "
              f"(removed {before - after} pre-attack rows)")
    else:
        print(f"  {fname}: unchanged")

print()

# ── STEP 2: LOAD DATA WITH FILE-BASED SPLIT ───────────────────────────────────
print("=" * 60)
print("STEP 2: Loading data (file-based train/test split)")
print("=" * 60)

train_dfs, test_dfs = [], []
for fpath in sorted(glob.glob(os.path.join(PROCESSED_DIR, "*.csv"))):
    fname = os.path.basename(fpath)
    df    = pd.read_csv(fpath)
    # _05 files go to test set (same split as original train_models.py)
    if "_05" in fname:
        test_dfs.append(df)
    else:
        train_dfs.append(df)

train_df = pd.concat(train_dfs, ignore_index=True).dropna(subset=FEATURES + ["label"])
test_df  = pd.concat(test_dfs,  ignore_index=True).dropna(subset=FEATURES + ["label"])

print(f"Train rows: {len(train_df)}")
print(f"Test  rows: {len(test_df)}")
print(f"Train label counts: {dict(train_df['label'].value_counts().sort_index())}")
print(f"Test  label counts: {dict(test_df['label'].value_counts().sort_index())}")
print()

# ── STEP 3: FIT SCALER ON TRAINING DATA ───────────────────────────────────────
scaler = StandardScaler()
scaler.fit(train_df[FEATURES].values)

with open(os.path.join(RESULTS_DIR, "lstm_scaler_v2.pkl"), "wb") as f:
    pickle.dump(scaler, f)
print(f"Scaler fitted and saved to {RESULTS_DIR}lstm_scaler_v2.pkl")
print()

# ── STEP 4: BUILD SEQUENCES PER FILE (preserves temporal order) ───────────────
print("=" * 60)
print("STEP 3: Building temporally-ordered sequences per file")
print("=" * 60)

def build_sequences(df_list, window_size):
    """
    Build LSTM input sequences keeping each window inside a single file.
    This ensures the model sees real temporal transitions (pre-attack -> onset
    -> full drift) rather than windows that straddle different flight scenarios.
    """
    Xs, ys = [], []
    for df in df_list:
        df_clean = df.dropna(subset=FEATURES + ["label"])
        if len(df_clean) < window_size:
            continue
        X = scaler.transform(df_clean[FEATURES].values)
        y = df_clean["label"].values.astype(int)
        for i in range(len(X) - window_size):
            Xs.append(X[i : i + window_size])
            ys.append(y[i + window_size - 1])   # label = last timestep in window
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int64)

X_train, y_train = build_sequences(train_dfs, WINDOW_SIZE)
X_test,  y_test  = build_sequences(test_dfs,  WINDOW_SIZE)

print(f"Train sequences shape: {X_train.shape}")
print(f"Test  sequences shape: {X_test.shape}")
print(f"Train seq label counts: {np.bincount(y_train).tolist()}")
print(f"Test  seq label counts: {np.bincount(y_test).tolist()}")
print()

# ── STEP 5: CLASS WEIGHTS ─────────────────────────────────────────────────────
counts  = np.bincount(y_train)
total   = counts.sum()
weights = torch.tensor(
    [total / (NUM_CLASSES * c) for c in counts], dtype=torch.float32
)
print(f"Class weights (Normal / Spoofing / Jamming): {weights.tolist()}")
print()

# ── STEP 6: DATALOADERS ───────────────────────────────────────────────────────
train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long)),
    batch_size=BATCH_SIZE, shuffle=True
)
test_loader = DataLoader(
    TensorDataset(torch.tensor(X_test), torch.tensor(y_test,  dtype=torch.long)),
    batch_size=BATCH_SIZE, shuffle=False
)

# ── STEP 7: MODEL ─────────────────────────────────────────────────────────────
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])   # classify from last timestep

device = torch.device("cpu")
model  = LSTMClassifier(
    len(FEATURES), HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES, DROPOUT
).to(device)

criterion = nn.CrossEntropyLoss(weight=weights.to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# ── STEP 8: TRAINING LOOP ─────────────────────────────────────────────────────
print("=" * 60)
print("STEP 4: Training")
print("=" * 60)

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out  = model(Xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(yb)
        correct    += (out.argmax(1) == yb).sum().item()
        total      += len(yb)
    scheduler.step()
    if epoch % 5 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}/{EPOCHS} | "
              f"Loss: {total_loss/total:.4f} | "
              f"Train Acc: {correct/total*100:.2f}%")

print()

# ── STEP 9: EVALUATION ────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 5: Evaluation on held-out test files (_05)")
print("=" * 60)

model.eval()
preds_all, labels_all = [], []
with torch.no_grad():
    for Xb, yb in test_loader:
        preds_all.extend(model(Xb.to(device)).argmax(1).cpu().numpy())
        labels_all.extend(yb.numpy())

preds_all  = np.array(preds_all)
labels_all = np.array(labels_all)

print(classification_report(
    labels_all, preds_all,
    target_names=["Normal", "Spoofing", "Jamming"],
    digits=4
))

cm = confusion_matrix(labels_all, preds_all)
print("Confusion matrix (rows = actual, cols = predicted):")
print(f"{'':15s} {'Normal':>8s} {'Spoofing':>10s} {'Jamming':>9s}")
for i, lbl in enumerate(["Normal", "Spoofing", "Jamming"]):
    print(f"  Actual {lbl:<10s}{cm[i][0]:>8d}{cm[i][1]:>10d}{cm[i][2]:>9d}")

# ── STEP 10: SAVE ─────────────────────────────────────────────────────────────
torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "lstm_model_v2.pt"))
print()
print(f"Model saved  -> {RESULTS_DIR}lstm_model_v2.pt")
print(f"Scaler saved -> {RESULTS_DIR}lstm_scaler_v2.pkl")
print()
print("Compare the Spoofing recall above against the previous 46%.")
print("If spoofing recall >= 70%, the fix worked.")
