"""
evaluate_spoofing_phases.py
===========================
Honest characterisation of spoofing detection for a GRADUAL drift attack.

Instead of a single recall number that averages over the (undetectable-by-
design) stealthy onset, this reports recall by ATTACK PHASE and estimates
DETECTION LATENCY — the standard way to evaluate a ramping attack, and a
metric your project brief explicitly lists.

Phases (rows where label==1 in each spoofing test file):
  - onset    : first ONSET_SEC seconds after attack start  (drift ~0, stealthy)
  - early    : ONSET_SEC .. ESTABLISHED_SEC                 (drift growing)
  - established : ESTABLISHED_SEC onward                    (full drift)

Reports, for BOTH Random Forest and LSTM:
  - overall spoofing recall (all attack rows)
  - recall within each phase
  - mean detection latency (seconds from attack start to first correct
    spoofing prediction, per test file)

Run from: ~/uav_security_project/ml_models/
    python3 evaluate_spoofing_phases.py

Reads the already-trained models:
    results/random_forest_model.pkl  + results/scaler.pkl
    results/lstm_real.pt             + results/lstm_real_scaler.pkl
"""

import os
import glob
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ── CONFIG ────────────────────────────────────────────────────────────────────
PROCESSED_DIR = '../data/processed/'
RESULTS_DIR   = '../results/'

ATTACK_START_SEC = 30.0      # spoofing attack begins here (per logger)
SAMPLE_HZ        = 10
ONSET_SEC        = 5.0       # 0-5s after start  = onset (stealthy)
ESTABLISHED_SEC  = 15.0      # >15s after start  = established (full drift)

WINDOW_SIZE = 20
HIDDEN_SIZE = 128
NUM_LAYERS  = 2
DROPOUT     = 0.3
NUM_CLASSES = 3

# Feature set used by the Random Forest (15 features)
RF_FEATURES = [
    'residual_lat', 'residual_lon', 'residual_alt', 'residual_mag',
    'residual_mag_mean', 'residual_mag_std', 'gps_fix_type', 'gps_satellites',
    'imu_acc_mag', 'baro_pressure', 'gps_drift_mag', 'gps_lat_drift',
    'gps_lon_drift', 'gps_imu_lat_diff', 'gps_imu_lon_diff'
]
# LSTM uses the same set
LSTM_FEATURES = RF_FEATURES


class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(len(LSTM_FEATURES), HIDDEN_SIZE, NUM_LAYERS,
                            batch_first=True, dropout=DROPOUT)
        self.head = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 64), nn.ReLU(),
            nn.Dropout(DROPOUT), nn.Linear(64, NUM_CLASSES))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def phase_of(row_time_in_attack):
    """Given seconds since attack start, return phase name."""
    if row_time_in_attack < ONSET_SEC:
        return 'onset'
    elif row_time_in_attack < ESTABLISHED_SEC:
        return 'early'
    return 'established'


def main():
    # ── Load the held-out spoofing test file (run 05) ──
    spoof_test_path = os.path.join(PROCESSED_DIR, 'gps_spoofing_05.csv')
    if not os.path.exists(spoof_test_path):
        print(f"Missing {spoof_test_path}")
        return
    df = pd.read_csv(spoof_test_path).dropna(subset=RF_FEATURES + ['label'])
    df = df.reset_index(drop=True)

    attack_start_row = int(ATTACK_START_SEC * SAMPLE_HZ)

    # ── RANDOM FOREST predictions ──
    with open(os.path.join(RESULTS_DIR, 'random_forest_model.pkl'), 'rb') as f:
        rf = pickle.load(f)
    with open(os.path.join(RESULTS_DIR, 'scaler.pkl'), 'rb') as f:
        rf_scaler = pickle.load(f)

    X_rf = rf_scaler.transform(df[RF_FEATURES].fillna(0).values)
    rf_pred = rf.predict(X_rf)   # one prediction per row

    # ── LSTM predictions (per-row, via trailing window) ──
    with open(os.path.join(RESULTS_DIR, 'lstm_real_scaler.pkl'), 'rb') as f:
        lstm_scaler = pickle.load(f)
    model = LSTMClassifier()
    model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, 'lstm_real.pt')))
    model.eval()

    X_lstm_scaled = lstm_scaler.transform(df[LSTM_FEATURES].fillna(0).values)
    lstm_pred = np.full(len(df), -1, dtype=int)   # -1 = no prediction (first window)
    seqs, idxs = [], []
    for i in range(len(X_lstm_scaled) - WINDOW_SIZE):
        seqs.append(X_lstm_scaled[i:i + WINDOW_SIZE])
        idxs.append(i + WINDOW_SIZE - 1)           # prediction aligns to last row
    if seqs:
        with torch.no_grad():
            out = model(torch.tensor(np.array(seqs, dtype=np.float32)))
            p = out.argmax(1).numpy()
        for j, row_idx in enumerate(idxs):
            lstm_pred[row_idx] = p[j]

    # ── Build phase labels for attack rows ──
    results = {}
    for model_name, pred in [('Random Forest', rf_pred), ('LSTM', lstm_pred)]:
        attack_mask = (df['label'] == 1).values
        phase_stats = {'onset': [0, 0], 'early': [0, 0], 'established': [0, 0]}
        overall = [0, 0]
        first_correct_row = None

        for i in range(len(df)):
            if not attack_mask[i]:
                continue
            if pred[i] == -1:
                continue  # LSTM warm-up rows with no prediction
            t_in_attack = (i - attack_start_row) / SAMPLE_HZ
            ph = phase_of(t_in_attack)
            correct = 1 if pred[i] == 1 else 0
            phase_stats[ph][0] += correct
            phase_stats[ph][1] += 1
            overall[0] += correct
            overall[1] += 1
            if correct and first_correct_row is None:
                first_correct_row = i

        latency = ((first_correct_row - attack_start_row) / SAMPLE_HZ
                   if first_correct_row is not None else None)
        results[model_name] = (overall, phase_stats, latency)

    # ── Report ──
    print("=" * 60)
    print("SPOOFING DETECTION BY ATTACK PHASE  (held-out run 05)")
    print("=" * 60)
    print(f"Attack starts at t={ATTACK_START_SEC:.0f}s | "
          f"onset<{ONSET_SEC:.0f}s | "
          f"early {ONSET_SEC:.0f}-{ESTABLISHED_SEC:.0f}s | "
          f"established >{ESTABLISHED_SEC:.0f}s\n")

    def pct(pair):
        c, n = pair
        return f"{(c/n*100):.1f}% ({c}/{n})" if n else "n/a"

    for model_name in ['Random Forest', 'LSTM']:
        overall, ph, latency = results[model_name]
        print(f"--- {model_name} ---")
        print(f"  Overall spoofing recall : {pct(overall)}")
        print(f"  Onset       (<{ONSET_SEC:.0f}s)   : {pct(ph['onset'])}")
        print(f"  Early    ({ONSET_SEC:.0f}-{ESTABLISHED_SEC:.0f}s)   : {pct(ph['early'])}")
        print(f"  Established (>{ESTABLISHED_SEC:.0f}s)  : {pct(ph['established'])}")
        if latency is not None:
            print(f"  Detection latency       : {latency:.1f}s after attack start")
        else:
            print(f"  Detection latency       : never detected")
        print()

    print("Interpretation: low onset recall is expected — the attack has not")
    print("meaningfully manifested yet. Established-phase recall is the honest")
    print("measure of detection capability once the spoof is active.")


if __name__ == '__main__':
    main()
