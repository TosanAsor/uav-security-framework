import numpy as np
import pandas as pd
import os

# ── CONFIGURATION ──
OUTPUT_DIR   = '../data/'
RUNS         = 5
DURATION     = 120
HZ           = 10
ATTACK_START = 30

os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_LAT = -35.363261
BASE_LON =  149.165230
BASE_ALT =  584.0

# Different drift rates per run — model must generalise
DRIFT_RATES = [0.0005, 0.0008, 0.0012, 0.0016, 0.002]

def generate_flight(run, attack_type):
    np.random.seed(run * 100 + attack_type)
    n_samples = DURATION * HZ
    t = np.linspace(0, DURATION, n_samples)

    # ── REALISTIC FLIGHT PATH ──
    flight_freq = 0.02
    lat = BASE_LAT + 0.0005 * np.sin(2 * np.pi * flight_freq * t)
    lon = BASE_LON + 0.0005 * np.sin(4 * np.pi * flight_freq * t)
    alt = BASE_ALT + 20 + 2 * np.sin(0.1 * t)

    # Realistic GPS noise
    lat += np.random.normal(0, 0.000005, n_samples)
    lon += np.random.normal(0, 0.000005, n_samples)
    alt += np.random.normal(0, 0.3, n_samples)

    # ── IMU ──
    imu_xacc  = np.diff(lat, prepend=lat[0]) * 1e6 + np.random.normal(0, 50, n_samples)
    imu_yacc  = np.diff(lon, prepend=lon[0]) * 1e6 + np.random.normal(0, 50, n_samples)
    imu_zacc  = -980 + np.random.normal(0, 30, n_samples)
    imu_xgyro = np.diff(lat, prepend=lat[0]) * 5e4 + np.random.normal(0, 20, n_samples)
    imu_ygyro = np.diff(lon, prepend=lon[0]) * 5e4 + np.random.normal(0, 20, n_samples)
    imu_zgyro = np.random.normal(0, 10, n_samples)

    # ── BARO ──
    baro_pressure = 950.0 - (alt - BASE_ALT) * 0.12 + np.random.normal(0, 0.05, n_samples)
    baro_temp     = 250 + np.random.normal(0, 1, n_samples)

    # ── MAG ──
    mag_x = 180 + np.random.normal(0, 2, n_samples)
    mag_y = -60  + np.random.normal(0, 2, n_samples)
    mag_z = 410  + np.random.normal(0, 2, n_samples)

    # ── GPS STATUS ──
    gps_fix_type   = np.full(n_samples, 6)
    gps_satellites = np.full(n_samples, 10, dtype=float)
    gps_speed      = np.sqrt(
        np.diff(lat, prepend=lat[0])**2 +
        np.diff(lon, prepend=lon[0])**2
    ) * 1e5

    labels     = np.zeros(n_samples, dtype=int)
    attack_idx = int(ATTACK_START * HZ)

    if attack_type == 1:  # GPS SPOOFING
        drift_rate = DRIFT_RATES[run - 1]
        # Random drift direction per run
        angle    = np.random.uniform(0, 2 * np.pi)
        dir_lat  = np.cos(angle)
        dir_lon  = np.sin(angle)

        for i in range(attack_idx, n_samples):
            elapsed  = t[i] - ATTACK_START
            drift    = drift_rate * elapsed
            # Add extra noise during spoofing to make it less perfect
            lat[i]   = lat[attack_idx - 1] + drift * dir_lat + np.random.normal(0, 0.00003)
            lon[i]   = lon[attack_idx - 1] + drift * dir_lon + np.random.normal(0, 0.00003)
            labels[i] = 1

    elif attack_type == 2:  # GPS JAMMING
        last_lat = lat[attack_idx - 1]
        last_lon = lon[attack_idx - 1]
        last_alt = alt[attack_idx - 1]
        for i in range(attack_idx, n_samples):
            lat[i]            = last_lat
            lon[i]            = last_lon
            alt[i]            = last_alt
            gps_fix_type[i]   = 0
            gps_satellites[i] = 0
            gps_speed[i]      = 0
            labels[i]         = 2

    df = pd.DataFrame({
        'timestamp':      np.round(t, 3),
        'gps_lat':        lat,
        'gps_lon':        lon,
        'gps_alt':        alt,
        'gps_fix_type':   gps_fix_type,
        'gps_satellites': gps_satellites,
        'gps_speed':      gps_speed,
        'imu_xacc':       imu_xacc,
        'imu_yacc':       imu_yacc,
        'imu_zacc':       imu_zacc,
        'imu_xgyro':      imu_xgyro,
        'imu_ygyro':      imu_ygyro,
        'imu_zgyro':      imu_zgyro,
        'baro_pressure':  baro_pressure,
        'baro_temp':      baro_temp,
        'mag_x':          mag_x,
        'mag_y':          mag_y,
        'mag_z':          mag_z,
        'label':          labels
    })

    return df

attack_names = {0: 'normal_flight', 1: 'gps_spoofing', 2: 'gps_jamming'}

for attack_type, name in attack_names.items():
    for run in range(1, RUNS + 1):
        df       = generate_flight(run, attack_type)
        filename = f'{name}_{run:02d}.csv'
        path     = os.path.join(OUTPUT_DIR, filename)
        df.to_csv(path, index=False)
        label_counts = df['label'].value_counts().to_dict()
        print(f"Generated: {filename} — {len(df)} rows — "
              f"labels: {label_counts}")

print(f"\nAll files generated in {OUTPUT_DIR}")
print(f"Total files: {RUNS * 3}")
