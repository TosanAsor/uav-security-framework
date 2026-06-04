"""
countermeasure.py
=================
Offline sensor-fallback countermeasure DEMONSTRATION.

This module replays a processed flight log row by row, runs the trained
Random Forest detector on each timestep, and implements the countermeasure
logic described in the project proposal:

  - SPOOFING detected -> stop trusting GPS, switch to IMU-only dead-reckoning
    for position. The module propagates an IMU-based position estimate from
    the last GPS position believed to be trustworthy.
  - JAMMING detected  -> raise a safe-land flag (GPS lost; controlled descent
    is the safe response).
  - NORMAL            -> trust GPS, keep IMU estimate synced to GPS.

To avoid reacting to single-frame misclassifications, detection is confirmed
over a short sliding window (N consecutive flags) before the response engages.

IMPORTANT — scope and honesty:
  This is an OFFLINE demonstration of the detection->response LOGIC on real
  logged SITL data. It is not a live flight-control system. It shows that,
  given a detection, the fallback estimate can be maintained from inertial
  data and that the response triggers correctly. Live, in-the-loop
  integration with the autopilot is future work (detection recall is moderate,
  so live triggering would miss/late-fire on some attacks).

USAGE:
    python3 countermeasure.py --file ../data/processed/gps_spoofing_05.csv
    python3 countermeasure.py --file ../data/processed/gps_jamming_05.csv
    python3 countermeasure.py --file ../data/processed/normal_flight_05.csv

Reads the trained model: results/random_forest_model.pkl + results/scaler.pkl
Writes a response log:    results/countermeasure_<flightname>.csv
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd

RESULTS_DIR = '../results/'

FEATURES = [
    'residual_lat', 'residual_lon', 'residual_alt', 'residual_mag',
    'residual_mag_mean', 'residual_mag_std', 'gps_fix_type', 'gps_satellites',
    'imu_acc_mag', 'baro_pressure', 'gps_drift_mag', 'gps_lat_drift',
    'gps_lon_drift', 'gps_imu_lat_diff', 'gps_imu_lon_diff'
]

CONFIRM_WINDOW = 5    # consecutive same-class flags needed to engage a response
LABEL_NAME = {0: 'NORMAL', 1: 'SPOOFING', 2: 'JAMMING'}


def load_model():
    with open(os.path.join(RESULTS_DIR, 'random_forest_model.pkl'), 'rb') as f:
        rf = pickle.load(f)
    with open(os.path.join(RESULTS_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    return rf, scaler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', required=True, help='processed flight CSV')
    ap.add_argument('--window', type=int, default=20, help='consecutive frames to confirm')
    args = ap.parse_args()

    global CONFIRM_WINDOW
    CONFIRM_WINDOW = args.window
    df = pd.read_csv(args.file).fillna(0).reset_index(drop=True)
    flight = os.path.splitext(os.path.basename(args.file))[0]

    rf, scaler = load_model()
    X = scaler.transform(df[FEATURES].values)
    preds = rf.predict(X)

    # ── State for the countermeasure ──
    mode = 'GPS'                 # GPS | IMU_FALLBACK | SAFE_LAND
    recent = []                  # rolling buffer of recent predictions
    # IMU dead-reckoning position estimate (lat/lon), seeded from first GPS fix
    est_lat = df['gps_lat'].iloc[0]
    est_lon = df['gps_lon'].iloc[0]

    out_rows = []
    engaged_row = None           # row index where fallback first engaged

    for i in range(len(df)):
        pred = int(preds[i])
        recent.append(pred)
        if len(recent) > CONFIRM_WINDOW:
            recent.pop(0)

        confirmed = (len(recent) == CONFIRM_WINDOW and len(set(recent)) == 1)
        confirmed_class = recent[-1] if confirmed else None

        # ── Position update ──
        if mode == 'GPS':
            # trust GPS; keep IMU estimate synced
            est_lat = df['gps_lat'].iloc[i]
            est_lon = df['gps_lon'].iloc[i]
        elif mode == 'IMU_FALLBACK':
            # dead reckon: propagate using IMU-derived per-step change,
            # ignoring the (untrusted) GPS
            est_lat += df['gps_lat_change'].iloc[i] * 0  # GPS ignored
            est_lat += df['imu_lat_change'].iloc[i]
            est_lon += df['imu_lon_change'].iloc[i]
        # SAFE_LAND: position hold at last trusted estimate (no update)

        # ── Mode transitions (only on confirmed detection) ──
        if confirmed:
            if confirmed_class == 1 and mode != 'IMU_FALLBACK':
                mode = 'IMU_FALLBACK'
                if engaged_row is None:
                    engaged_row = i
            elif confirmed_class == 2 and mode != 'SAFE_LAND':
                mode = 'SAFE_LAND'
                if engaged_row is None:
                    engaged_row = i
            elif confirmed_class == 0 and mode != 'GPS':
                # sustained normal -> recover to GPS trust
                mode = 'GPS'

        out_rows.append({
            'timestamp': df['timestamp'].iloc[i],
            'true_label': int(df['label'].iloc[i]),
            'predicted': pred,
            'predicted_name': LABEL_NAME[pred],
            'mode': mode,
            'gps_lat': df['gps_lat'].iloc[i],
            'gps_lon': df['gps_lon'].iloc[i],
            'est_lat': est_lat,
            'est_lon': est_lon,
        })

    out = pd.DataFrame(out_rows)
    out_path = os.path.join(RESULTS_DIR, f'countermeasure_{flight}.csv')
    out.to_csv(out_path, index=False)

    # ── Report ──
    print("=" * 60)
    print(f"COUNTERMEASURE REPLAY — {flight}")
    print("=" * 60)
    n = len(out)
    print(f"Rows processed: {n}")
    print(f"Confirmation window: {CONFIRM_WINDOW} consecutive frames\n")

    # time spent in each mode
    print("Time in each mode:")
    for m in ['GPS', 'IMU_FALLBACK', 'SAFE_LAND']:
        c = (out['mode'] == m).sum()
        print(f"  {m:13s}: {c:5d} rows ({c/n*100:5.1f}%)")
    print()

    # when did the response engage, relative to true attack onset?
    attack_rows = out[out['true_label'] != 0]
    if len(attack_rows) and engaged_row is not None:
        attack_start = attack_rows.index[0]
        t_attack = out['timestamp'].iloc[attack_start]
        t_engage = out['timestamp'].iloc[engaged_row]
        delay = t_engage - t_attack
        print(f"True attack onset : t={t_attack:.1f}s (row {attack_start})")
        print(f"Response engaged  : t={t_engage:.1f}s (row {engaged_row})")
        print(f"Response delay    : {delay:.1f}s after attack onset")
    elif engaged_row is None:
        print("Response never engaged (no class confirmed over the window).")
    print()

    # how far did the GPS diverge from the IMU fallback estimate during fallback?
    fb = out[out['mode'] == 'IMU_FALLBACK']
    if len(fb):
        gap = np.sqrt((fb['gps_lat'] - fb['est_lat'])**2 +
                      (fb['gps_lon'] - fb['est_lon'])**2)
        print("During IMU fallback, divergence between (spoofed) GPS and "
              "the IMU-only estimate:")
        print(f"  mean {gap.mean():.6f} deg, max {gap.max():.6f} deg")
        print("  (a growing gap = the fallback is rejecting the spoofed drift)")
    print()
    print(f"Response log written to: {out_path}")


if __name__ == '__main__':
    main()
