import pandas as pd
import numpy as np
import glob
import os

PROCESSED_DIR = '../data/processed/'

all_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, '*.csv')))
dfs = []
for f in all_files:
    df = pd.read_csv(f)
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

normal   = data[data['label'] == 0]
spoofing = data[data['label'] == 1]
jamming  = data[data['label'] == 2]

print("=" * 60)
print("RESIDUAL COMPARISON BY CLASS")
print("=" * 60)

for col in ['residual_lat', 'residual_lon', 'residual_mag',
            'residual_mag_mean', 'residual_mag_std']:
    print(f"\n{col}:")
    print(f"  Normal   — mean: {normal[col].mean():.6f}  std: {normal[col].std():.6f}")
    print(f"  Spoofing — mean: {spoofing[col].mean():.6f}  std: {spoofing[col].std():.6f}")
    print(f"  Jamming  — mean: {jamming[col].mean():.6f}  std: {jamming[col].std():.6f}")

print("\n" + "=" * 60)
print("DRIFT FEATURES BY CLASS")
print("=" * 60)

for col in ['gps_lat_drift', 'gps_lon_drift', 'gps_drift_mag',
            'gps_imu_lat_diff', 'gps_imu_lon_diff']:
    print(f"\n{col}:")
    print(f"  Normal   — mean: {normal[col].mean():.8f}  std: {normal[col].std():.8f}")
    print(f"  Spoofing — mean: {spoofing[col].mean():.8f}  std: {spoofing[col].std():.8f}")
    print(f"  Jamming  — mean: {jamming[col].mean():.8f}  std: {jamming[col].std():.8f}")
