"""
train_lstm_real.py
==================
LSTM classifier for the real SITL dataset (normal / spoofing / jamming).

Design choices, each defensible in a viva:
  - Sequence model (20-step windows) — suited to GRADUAL spoofing drift
    that a single-row classifier (Random Forest) cannot see.
  - Per-file sequence construction — windows never straddle two flights,
    so temporal context is real (pre-attack -> onset -> drift stays intact).
  - File-based train/test split — runs 01-04 train, run 05 tests. No row
    from a test flight appears in training (prevents leakage).
  - Class weighting — counters the normal-heavy class balance without
    inventing or duplicating data.
  - All 15 EKF/drift features (same set as the Random Forest) so the two
    models are compared on equal footing.

Run from: ~/uav_security_project/ml_models/
    python3 train_lstm_real.py
"""

import os
import glob
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── CONFIG ────────────────────────────────────────────────────────────────────
PROCESSED_DIR = '../data/processed/'
RESULTS_DIR   = '../results/'
os.makedirs(RESULTS_DIR, exist_ok=True)

WINDOW_SIZE   = 20
HIDDEN_SIZE   = 128
NUM_LAYERS    = 2
DROPOUT       = 0.3
BATCH_SIZE    = 64
EPOCHS        = 40
LR            = 1e-3
NUM_CLASSES   = 3
SEED          = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

FEATURES = [
    'residual_lat', 'residual_lon', 'residual_alt', 'residual_mag',
    'residual_mag_mean', 'residual_mag_std', 'gps_fix_type', 'gps_satellites',
    'imu_acc_mag', 'baro_pressure', 'gps_drift_mag', 'gps_lat_drift',
    'gps_lon_drift', 'gps_imu_lat_diff', 'gps_imu_lon_diff'
]

CLASS_NAMES = ['Normal', 'Spoofing', 'Jamming']


# ── DATA LOADING ──────────────────────────────────────────────────────────────
def load_split():
    """Return (train_dfs, test_dfs). Run 05 of each class -> test."""
    train_dfs, test_dfs = [], []
    for path in sorted(glob.glob(os.path.join(PROCESSED_DIR, '*.csv'))):
        name = os.path.basename(path)
        df = pd.read_csv(path).dropna(subset=FEATURES + ['label'])
        if '_05' in name:
            test_dfs.append((name, df))
        else:
            train_dfs.append((name, df))
    return train_dfs, test_dfs


def build_sequences(df_list, scaler):
    """
    Build LSTM windows within each file (never across files).
    Label of a window = label of its LAST timestep.
    """
    X_all, y_all = [], []
    for _, df in df_list:
        if len(df) < WINDOW_SIZE:
            continue
        X = scaler.transform(df[FEATURES].values)
        y = df['label'].values.astype(int)
        for i in range(len(X) - WINDOW_SIZE):
            X_all.append(X[i:i + WINDOW_SIZE])
            y_all.append(y[i + WINDOW_SIZE - 1])
    return (np.array(X_all, dtype=np.float32),
            np.array(y_all, dtype=np.int64))


# ── MODEL ─────────────────────────────────────────────────────────────────────
class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            len(FEATURES), HIDDEN_SIZE, NUM_LAYERS,
            batch_first=True, dropout=DROPOUT
        )
        self.head = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, NUM_CLASSES)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading data (file-based split: runs 01-04 train, run 05 test)...")
    train_dfs, test_dfs = load_split()
    print(f"  Train files: {len(train_dfs)}  |  Test files: {len(test_dfs)}")

    # Fit scaler on training rows only (no leakage from test)
    train_concat = pd.concat([df for _, df in train_dfs], ignore_index=True)
    scaler = StandardScaler().fit(train_concat[FEATURES].values)

    X_train, y_train = build_sequences(train_dfs, scaler)
    X_test,  y_test  = build_sequences(test_dfs,  scaler)

    print(f"  Train sequences: {X_train.shape}  labels {np.bincount(y_train).tolist()}")
    print(f"  Test  sequences: {X_test.shape}  labels {np.bincount(y_test).tolist()}")

    # Class weights from training distribution
    counts = np.bincount(y_train, minlength=NUM_CLASSES)
    weights = torch.tensor(
        [counts.sum() / (NUM_CLASSES * c) if c else 0.0 for c in counts],
        dtype=torch.float32
    )
    print(f"  Class weights: {weights.tolist()}")

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
        batch_size=BATCH_SIZE, shuffle=False
    )

    device = torch.device('cpu')
    model  = LSTMClassifier().to(device)
    crit   = nn.CrossEntropyLoss(weight=weights.to(device))
    opt    = torch.optim.Adam(model.parameters(), lr=LR)
    sched  = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)

    print("\n" + "=" * 55)
    print("TRAINING LSTM")
    print("=" * 55)
    losses, accs = [], []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tot_loss, correct, total = 0.0, 0, 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(Xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(yb)
            correct  += (out.argmax(1) == yb).sum().item()
            total    += len(yb)
        sched.step()
        losses.append(tot_loss / total)
        accs.append(correct / total)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS}  loss {losses[-1]:.4f}  "
                  f"train_acc {accs[-1]*100:.2f}%")

    # ── EVALUATE ──
    print("\n" + "=" * 55)
    print("EVALUATION (held-out run 05)")
    print("=" * 55)
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            preds.extend(model(Xb.to(device)).argmax(1).cpu().numpy())
            labels.extend(yb.numpy())
    preds, labels = np.array(preds), np.array(labels)

    present = sorted(np.unique(labels))
    names   = [CLASS_NAMES[i] for i in present]
    print(classification_report(labels, preds, labels=present,
          target_names=names, digits=4, zero_division=0))

    cm = confusion_matrix(labels, preds, labels=present)
    print("Confusion matrix (rows=actual, cols=predicted):")
    print("        " + "  ".join(f"{n[:7]:>7}" for n in names))
    for i, n in enumerate(names):
        print(f"  {n[:7]:>7} " + "  ".join(f"{v:>7}" for v in cm[i]))

    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    print(f"\nMacro F1: {macro_f1:.4f}")

    # spoofing recall callout (your primary attack)
    if 1 in present:
        idx = present.index(1)
        spoof_recall = cm[idx, idx] / cm[idx].sum() if cm[idx].sum() else 0
        print(f"Spoofing recall (PRIMARY): {spoof_recall*100:.2f}%")

    # ── SAVE ──
    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, 'lstm_real.pt'))
    with open(os.path.join(RESULTS_DIR, 'lstm_real_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    # training curve
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(losses, color='steelblue'); a1.set_title('Training Loss'); a1.set_xlabel('Epoch')
    a2.plot([a*100 for a in accs], color='green'); a2.set_title('Training Accuracy (%)'); a2.set_xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'lstm_real_training.png'), dpi=150)
    plt.close()

    print(f"\nSaved: results/lstm_real.pt, results/lstm_real_scaler.pkl, "
          f"results/lstm_real_training.png")


if __name__ == '__main__':
    main()
