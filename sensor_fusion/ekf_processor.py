import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter
import os
import glob

DATA_DIR      = '../data/'
OUTPUT_DIR    = '../data/processed/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

    kf.R  = np.eye(3) * 0.0001
    kf.Q  = np.eye(6) * 0.000001
    kf.P  = np.eye(6) * 1.0

    return kf

def process_file(filepath):
    filename = os.path.basename(filepath)
    print(f"Processing: {filename}")

    df = pd.read_csv(filepath)
    df = df[~((df['gps_fix_type'] == 0) & (df['label'] != 2))].copy()
    df = df.reset_index(drop=True)

    if len(df) < 10:
        print(f"  Skipping — not enough rows")
        return

    valid_rows = df[df['gps_fix_type'] > 0]
    if len(valid_rows) == 0:
        init_lat = -35.363261
        init_lon =  149.165230
        init_alt =  584.0
    else:
        init_lat = valid_rows['gps_lat'].iloc[0]
        init_lon = valid_rows['gps_lon'].iloc[0]
        init_alt = valid_rows['gps_alt'].iloc[0]

    kf = create_ekf()
    kf.x = np.array([init_lat, init_lon, init_alt, 0.0, 0.0, 0.0])

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

        obs_lat = row['gps_lat']
        obs_lon = row['gps_lon']
        obs_alt = row['gps_alt']

        z = np.array([obs_lat, obs_lon, obs_alt])

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

    # ── CUMULATIVE DRIFT FEATURES ──
    # How far has GPS moved from where it was 30 steps ago
    window = 30
    df['gps_lat_drift'] = (
        df['gps_lat'] - df['gps_lat'].shift(window)
    ).fillna(0).abs()
    df['gps_lon_drift'] = (
        df['gps_lon'] - df['gps_lon'].shift(window)
    ).fillna(0).abs()
    df['gps_drift_mag'] = np.sqrt(
        df['gps_lat_drift']**2 + df['gps_lon_drift']**2
    )

    # ── GPS vs IMU CONSISTENCY ──
    # IMU-predicted displacement vs GPS displacement
    df['imu_lat_change'] = (
        df['imu_xacc'].rolling(10, min_periods=1).mean() * 0.0000001
    )
    df['imu_lon_change'] = (
        df['imu_yacc'].rolling(10, min_periods=1).mean() * 0.0000001
    )
    df['gps_lat_change'] = df['gps_lat'].diff().fillna(0)
    df['gps_lon_change'] = df['gps_lon'].diff().fillna(0)

    df['gps_imu_lat_diff'] = (df['gps_lat_change'] - df['imu_lat_change']).abs()
    df['gps_imu_lon_diff'] = (df['gps_lon_change'] - df['imu_lon_change']).abs()





    out_path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(out_path, index=False)

    label_counts = df['label'].value_counts().to_dict()
    avg_residual = df['residual_mag'].mean()
    avg_drift    = df['gps_drift_mag'].mean()
    print(f"  Done — {len(df)} rows — labels: {label_counts} — "
          f"avg residual: {avg_residual:.6f} — "
          f"avg drift: {avg_drift:.6f}")

all_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.csv')))
print(f"Found {len(all_files)} files to process\n")

for f in all_files:
    process_file(f)

print("\nAll files processed.")
print(f"Saved to: {OUTPUT_DIR}")
