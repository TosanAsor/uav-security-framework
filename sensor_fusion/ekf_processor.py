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
    """
    State vector: [lat, lon, alt, vel_lat, vel_lon, vel_alt]
    We track position and velocity in 3 dimensions.
    """
    kf = KalmanFilter(dim_x=6, dim_z=3)

    dt = 0.1  # 10 Hz sampling

    # State transition matrix — constant velocity model
    kf.F = np.array([
        [1, 0, 0, dt, 0,  0 ],
        [0, 1, 0, 0,  dt, 0 ],
        [0, 0, 1, 0,  0,  dt],
        [0, 0, 0, 1,  0,  0 ],
        [0, 0, 0, 0,  1,  0 ],
        [0, 0, 0, 0,  0,  1 ],
    ])

    # Measurement matrix — we observe position only (lat, lon, alt)
    kf.H = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
    ])

    # Measurement noise — how much we trust GPS
    kf.R = np.eye(3) * 0.0001

    # Process noise — how much the state can change between steps
    kf.Q = np.eye(6) * 0.00001

    # Initial state covariance
    kf.P = np.eye(6) * 1.0

    return kf

# ── PROCESS ONE FILE ──
def process_file(filepath):
    filename = os.path.basename(filepath)
    print(f"Processing: {filename}")

    df = pd.read_csv(filepath)

    # Skip rows where GPS has no fix yet
    df = df[df['gps_fix_type'] > 0].copy()
    df = df.reset_index(drop=True)

    if len(df) < 10:
        print(f"  Skipping — not enough rows with GPS fix")
        return

    # Create EKF
    kf = create_ekf()

    # Initialise state from first GPS reading
    kf.x = np.array([
        df['gps_lat'].iloc[0],
        df['gps_lon'].iloc[0],
        df['gps_alt'].iloc[0],
        0.0, 0.0, 0.0  # initial velocity = 0
    ])

    # ── FEATURE ARRAYS ──
    residual_lat  = []
    residual_lon  = []
    residual_alt  = []
    residual_mag  = []  # magnitude of position residual
    predicted_lat = []
    predicted_lon = []
    predicted_alt = []

    for i, row in df.iterrows():
        # Predict next state using EKF
        kf.predict()

        # Store prediction before update
        pred_lat = kf.x[0]
        pred_lon = kf.x[1]
        pred_alt = kf.x[2]

        # GPS observation
        z = np.array([row['gps_lat'], row['gps_lon'], row['gps_alt']])

        # Update EKF with GPS measurement
        kf.update(z)

        # Compute residual (GPS - prediction)
        res_lat = row['gps_lat'] - pred_lat
        res_lon = row['gps_lon'] - pred_lon
        res_alt = row['gps_alt'] - pred_alt
        res_mag = np.sqrt(res_lat**2 + res_lon**2 + res_alt**2)

        residual_lat.append(res_lat)
        residual_lon.append(res_lon)
        residual_alt.append(res_alt)
        residual_mag.append(res_mag)
        predicted_lat.append(pred_lat)
        predicted_lon.append(pred_lon)
        predicted_alt.append(pred_alt)

    # Add new feature columns
    df['residual_lat']  = residual_lat
    df['residual_lon']  = residual_lon
    df['residual_alt']  = residual_alt
    df['residual_mag']  = residual_mag
    df['predicted_lat'] = predicted_lat
    df['predicted_lon'] = predicted_lon
    df['predicted_alt'] = predicted_alt

    # Also add rolling statistics as features
    df['residual_mag_mean'] = df['residual_mag'].rolling(10, min_periods=1).mean()
    df['residual_mag_std']  = df['residual_mag'].rolling(10, min_periods=1).std().fillna(0)
    df['imu_acc_mag']       = np.sqrt(df['imu_xacc']**2 + df['imu_yacc']**2 + df['imu_zacc']**2)

    # Save processed file
    out_path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(out_path, index=False)

    avg_residual = df['residual_mag'].mean()
    max_residual = df['residual_mag'].max()
    print(f"  Done — {len(df)} rows — avg residual: {avg_residual:.6f} — max residual: {max_residual:.6f}")
    print(f"  Saved to: {out_path}")

# ── PROCESS ALL FILES ──
all_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.csv')))
print(f"Found {len(all_files)} files to process\n")

for f in all_files:
    process_file(f)

print("\nAll files processed.")
print(f"Processed files saved to: {OUTPUT_DIR}")
