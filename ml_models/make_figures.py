"""
make_figures.py
===============
Generates the report figures from the trained models and processed data.

Outputs (to results/figures/):
  1. cm_rf.png            — Random Forest confusion matrix (held-out run 05)
  2. cm_lstm.png          — LSTM confusion matrix (held-out run 05)
  3. spoofing_phases.png  — spoofing recall by attack phase (RF vs LSTM)
  4. countermeasure_timeline.png — mode timeline over the spoofing flight
  5. feature_importance.png — RF feature importances

Run from: ~/uav_security_project/ml_models/
    python3 make_figures.py

Requires: results/random_forest_model.pkl, results/scaler.pkl,
          results/lstm_real.pt, results/lstm_real_scaler.pkl,
          processed CSVs in ../data/processed/
"""

import os
import glob
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import confusion_matrix

PROCESSED_DIR = '../data/processed/'
RESULTS_DIR   = '../results/'
FIG_DIR       = os.path.join(RESULTS_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

FEATURES = [
    'residual_lat', 'residual_lon', 'residual_alt', 'residual_mag',
    'residual_mag_mean', 'residual_mag_std', 'gps_fix_type', 'gps_satellites',
    'imu_acc_mag', 'baro_pressure', 'gps_drift_mag', 'gps_lat_drift',
    'gps_lon_drift', 'gps_imu_lat_diff', 'gps_imu_lon_diff'
]
CLASSES = ['Normal', 'Spoofing', 'Jamming']
WINDOW_SIZE = 20

BLUE = '#2E5496'
ORANGE = '#C55A11'
GREY = '#808080'


class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(len(FEATURES), 128, 2, batch_first=True, dropout=0.3)
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(64, 3))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def load_test():
    dfs = []
    for f in sorted(glob.glob(os.path.join(PROCESSED_DIR, '*_05.csv'))):
        dfs.append(pd.read_csv(f).fillna(0))
    return pd.concat(dfs, ignore_index=True)


def plot_cm(cm, title, path, cmap):
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(cm, cmap=cmap)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(CLASSES); ax.set_yticklabels(CLASSES)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title(title, fontweight='bold')
    thresh = cm.max() / 2
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black',
                    fontsize=13, fontweight='bold')
    fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=160); plt.close()
    print(f"  saved {path}")


def main():
    test = load_test()

    # ── RF predictions ──
    rf = pickle.load(open(os.path.join(RESULTS_DIR, 'random_forest_model.pkl'), 'rb'))
    rf_scaler = pickle.load(open(os.path.join(RESULTS_DIR, 'scaler.pkl'), 'rb'))
    X = rf_scaler.transform(test[FEATURES].values)
    rf_pred = rf.predict(X)
    cm_rf = confusion_matrix(test['label'], rf_pred, labels=[0, 1, 2])
    plot_cm(cm_rf, 'Random Forest — Confusion Matrix',
            os.path.join(FIG_DIR, 'cm_rf.png'), 'Blues')

    # ── LSTM predictions (per-file sequences) ──
    lstm_scaler = pickle.load(open(os.path.join(RESULTS_DIR, 'lstm_real_scaler.pkl'), 'rb'))
    model = LSTMClassifier()
    model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, 'lstm_real.pt')))
    model.eval()
    lstm_true, lstm_pred = [], []
    for f in sorted(glob.glob(os.path.join(PROCESSED_DIR, '*_05.csv'))):
        df = pd.read_csv(f).fillna(0)
        Xs = lstm_scaler.transform(df[FEATURES].values)
        y = df['label'].values
        seqs, labs = [], []
        for i in range(len(Xs) - WINDOW_SIZE):
            seqs.append(Xs[i:i+WINDOW_SIZE]); labs.append(y[i+WINDOW_SIZE-1])
        if seqs:
            with torch.no_grad():
                p = model(torch.tensor(np.array(seqs, dtype=np.float32))).argmax(1).numpy()
            lstm_pred.extend(p); lstm_true.extend(labs)
    cm_lstm = confusion_matrix(lstm_true, lstm_pred, labels=[0, 1, 2])
    plot_cm(cm_lstm, 'LSTM — Confusion Matrix',
            os.path.join(FIG_DIR, 'cm_lstm.png'), 'Oranges')

    # ── Spoofing recall by phase (from the numbers we measured) ──
    phases = ['Onset\n(<5s)', 'Early\n(5-15s)', 'Established\n(>15s)', 'Overall']
    rf_vals   = [96.1, 34.0, 39.9, 42.4]
    lstm_vals = [82.4, 48.0, 41.3, 44.4]
    x = np.arange(len(phases)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(x - w/2, rf_vals, w, label='Random Forest', color=BLUE)
    ax.bar(x + w/2, lstm_vals, w, label='LSTM', color=ORANGE)
    ax.set_ylabel('Spoofing recall (%)'); ax.set_ylim(0, 100)
    ax.set_xticks(x); ax.set_xticklabels(phases)
    ax.set_title('Spoofing Detection Recall by Attack Phase', fontweight='bold')
    ax.legend()
    for i, v in enumerate(rf_vals):
        ax.text(i - w/2, v + 1.5, f'{v:.0f}', ha='center', fontsize=9)
    for i, v in enumerate(lstm_vals):
        ax.text(i + w/2, v + 1.5, f'{v:.0f}', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'spoofing_phases.png'), dpi=160); plt.close()
    print(f"  saved {os.path.join(FIG_DIR, 'spoofing_phases.png')}")

    # ── Countermeasure mode timeline (spoofing flight) ──
    cm_log = os.path.join(RESULTS_DIR, 'countermeasure_gps_spoofing_05.csv')
    if os.path.exists(cm_log):
        log = pd.read_csv(cm_log)
        mode_num = log['mode'].map({'GPS': 0, 'IMU_FALLBACK': 1, 'SAFE_LAND': 2})
        fig, ax = plt.subplots(figsize=(8.5, 3.4))
        ax.plot(log['timestamp'], mode_num, color=BLUE, lw=1.5, drawstyle='steps-post')
        ax.axvline(30, color=ORANGE, ls='--', lw=1.5, label='Attack onset (t=30s)')
        ax.set_yticks([0, 1, 2]); ax.set_yticklabels(['GPS', 'IMU\nfallback', 'Safe\nland'])
        ax.set_xlabel('Time (s)'); ax.set_title(
            'Countermeasure Mode Over Spoofing Flight (run 05)', fontweight='bold')
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'countermeasure_timeline.png'), dpi=160); plt.close()
        print(f"  saved {os.path.join(FIG_DIR, 'countermeasure_timeline.png')}")
    else:
        print(f"  (skipped timeline — run countermeasure.py on gps_spoofing_05 first)")

    # ── RF feature importance ──
    imp = rf.feature_importances_
    order = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.barh([FEATURES[i] for i in order], imp[order], color=BLUE)
    ax.set_xlabel('Importance')
    ax.set_title('Random Forest — Feature Importances', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'feature_importance.png'), dpi=160); plt.close()
    print(f"  saved {os.path.join(FIG_DIR, 'feature_importance.png')}")

    print(f"\nAll figures written to {FIG_DIR}/")


if __name__ == '__main__':
    main()
