import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter
import os
import glob

# ── CONFIGURATION ──
DATA_DIR      = '../data/'
OUTPUT_DIR    = '../data/processed/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── EKF SETUP ──
def create_ekf():
    kf = KalmanFilter(dim_x=6, dim_z=3)
    dt = 0.1

    kf.F = np.array([
        [1, 0, 0, dt, 0,  0 ],
        [0, 1, 0, 0,  dt, 0 ],
        [0, 0, 1, 0,  0,  dt],
        [0, 0, 0, 1,  0,  0 ],
        [0, 0, 0, 0,  1,  0 ],
        [0, 0, 0, 0,  0,  1 ],
    ])

    kf.H = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
    ])

    kf.R = np.eye(3) * 0.0001
    kf.Q = np.eye(6) * 0.00001
    kf.P = np.eye(6) * 1.0

    return kf

# ── PROCESS ONE FILE ──
def process_file(filepath):
    filename = os.path.basename(filepath)
    print(f"Processing: {filename}")

    df = pd.read_csv(filepath)

    # Keep jamming rows (label=2, fix=0) but remove GPS acquisition rows
    # (label=0 or 1, fix=0) which are just startup noise
    df = df[~((df['gps_fix_type'] == 0) & (df['label'] != 2))].copy()
    df = df.reset_index(drop=True)

    if len(df) < 10:
        print(f"  Skipping — not enough rows")
        return

    # For EKF initialisation we need a valid starting position
    # Use last known good GPS position before jamming starts
    valid_rows = df[df['gps_fix_type'] > 0]
    if len(valid_rows) == 0:
        # All rows are jamming — use home position
        init_lat = -35.363261
        init_lon = 149.165230
        init_alt = 584.0
    else:
        init_lat = valid_rows['gps_lat'].iloc[0]
        init_lon = valid_rows['gps_lon'].iloc[0]
        init_alt = valid_rows['gps_alt'].iloc[0]

    # Create EKF
    kf = create_ekf()
    kf.x = np.array([init_lat, init_lon, init_alt, 0.0, 0.0, 0.0])

    # ── FEATURE ARRAYS ──
    residual_lat  = []
    residual_lon  = []
    residual_alt  = []
    residual_mag  = []
    predicted_lat = []
    predicted_lon = []
    predicted_alt = []

    for i, row in df.iterrows():
        kf.predict()

        pred_lat = kf.x[0]
        pred_lon = kf.x[1]
        pred_alt = kf.x[2]

        # Use frozen last-known position for jamming rows
        obs_lat = row['gps_lat']
        obs_lon = row['gps_lon']
        obs_alt = row['gps_alt']

        z = np.array([obs_lat, obs_lon, obs_alt])

        # Only update EKF with real GPS (not jammed/frozen readings)
        if row['gps_fix_type'] > 0:
            kf.update(z)

        res_lat = obs_lat - pred_lat
        res_lon = obs_lon - pred_lon
        res_alt = obs_alt - pred_alt
        res_mag = np.sqrt(res_lat**2 + res_lon**2 + res_alt**2)

        residual_lat.append(res_lat)
        residual_lon.append(res_lon)
        residual_alt.append(res_alt)
        residual_mag.append(res_mag)
        predicted_lat.append(pred_lat)
        predicted_lon.append(pred_lon)
        predicted_alt.append(pred_alt)

    df['residual_lat']  = residual_lat
    df['residual_lon']  = residual_lon
    df['residual_alt']  = residual_alt
    df['residual_mag']  = residual_mag
    df['predicted_lat'] = predicted_lat
    df['predicted_lon'] = predicted_lon
    df['predicted_alt'] = predicted_alt

    df['residual_mag_mean'] = df['residual_mag'].rolling(10, min_periods=1).mean()
    df['residual_mag_std']  = df['residual_mag'].rolling(10, min_periods=1).std().fillna(0)
    df['imu_acc_mag']       = np.sqrt(
        df['imu_xacc']**2 + df['imu_yacc']**2 + df['imu_zacc']**2
    )

    out_path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(out_path, index=False)

    label_counts = df['label'].value_counts().to_dict()
    avg_residual = df['residual_mag'].mean()
    max_residual = df['residual_mag'].max()
    print(f"  Done — {len(df)} rows — labels: {label_counts} — avg residual: {avg_residual:.6f} — max: {max_residual:.6f}")

# ── PROCESS ALL FILES ──
all_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.csv')))
print(f"Found {len(all_files)} files to process\n")

for f in all_files:
    process_file(f)

print("\nAll files processed.")
print(f"Saved to: {OUTPUT_DIR}")
