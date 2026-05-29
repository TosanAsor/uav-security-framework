from pymavlink import mavutil
import csv
import time
import os

# ── CONFIGURATION ──
OUTPUT_FILE     = '../data/gps_spoofing_01.csv'
FLIGHT_DURATION = 120
ATTACK_START    = 30
LABEL_NORMAL    = 0
LABEL_SPOOFING  = 1

# ── SPOOF SETTINGS ──
BASE_LAT   = -35.363261
BASE_LON   = 149.165230
BASE_ALT   = 584.0
DRIFT_RATE = 0.00005  # degrees per second

# ── CONNECT ──
print("Connecting to SITL...")
master = mavutil.mavlink_connection('tcp:127.0.0.1:5760')
master.wait_heartbeat()
print("Connected.")

master.mav.request_data_stream_send(
    master.target_system,
    master.target_component,
    mavutil.mavlink.MAV_DATA_STREAM_ALL,
    10, 1
)
time.sleep(2)

# ── PREPARE CSV ──
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
fields = [
    'timestamp',
    'gps_lat', 'gps_lon', 'gps_alt',
    'gps_fix_type', 'gps_satellites', 'gps_speed',
    'imu_xacc', 'imu_yacc', 'imu_zacc',
    'imu_xgyro', 'imu_ygyro', 'imu_zgyro',
    'baro_pressure', 'baro_temp',
    'mag_x', 'mag_y', 'mag_z',
    'label'
]

sensors = {
    'gps_lat': 0, 'gps_lon': 0, 'gps_alt': 0,
    'gps_fix_type': 0, 'gps_satellites': 0, 'gps_speed': 0,
    'imu_xacc': 0, 'imu_yacc': 0, 'imu_zacc': 0,
    'imu_xgyro': 0, 'imu_ygyro': 0, 'imu_zgyro': 0,
    'baro_pressure': 0, 'baro_temp': 0,
    'mag_x': 0, 'mag_y': 0, 'mag_z': 0,
}

print(f"Logging to {OUTPUT_FILE}")
print(f"Attack starts at t={ATTACK_START}s")
print("During spoofing: GPS coords replaced with drifting fake values\n")

start_time   = time.time()
rows_written = 0

with open(OUTPUT_FILE, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fields)
    writer.writeheader()

    while True:
        elapsed = time.time() - start_time
        if elapsed > FLIGHT_DURATION:
            break

        msg = master.recv_match(
            type=['GPS_RAW_INT', 'RAW_IMU', 'SCALED_PRESSURE'],
            blocking=True,
            timeout=1
        )

        if msg is None:
            continue

        msg_type = msg.get_type()

        if msg_type == 'GPS_RAW_INT':
            # Always read real GPS first
            sensors['gps_lat']        = msg.lat / 1e7
            sensors['gps_lon']        = msg.lon / 1e7
            sensors['gps_alt']        = msg.alt / 1000.0
            sensors['gps_fix_type']   = msg.fix_type
            sensors['gps_satellites'] = msg.satellites_visible
            sensors['gps_speed']      = msg.vel / 100.0

            # During spoofing phase — overwrite GPS coords with fake drifting ones
            if elapsed >= ATTACK_START:
                drift = DRIFT_RATE * (elapsed - ATTACK_START)
                sensors['gps_lat'] = BASE_LAT + drift
                sensors['gps_lon'] = BASE_LON + drift
                label = LABEL_SPOOFING
            else:
                label = LABEL_NORMAL

        elif msg_type == 'RAW_IMU':
            # IMU always stays real — this is the mismatch the EKF detects
            sensors['imu_xacc']  = msg.xacc
            sensors['imu_yacc']  = msg.yacc
            sensors['imu_zacc']  = msg.zacc
            sensors['imu_xgyro'] = msg.xgyro
            sensors['imu_ygyro'] = msg.ygyro
            sensors['imu_zgyro'] = msg.zgyro
            sensors['mag_x']     = msg.xmag
            sensors['mag_y']     = msg.ymag
            sensors['mag_z']     = msg.zmag

        elif msg_type == 'SCALED_PRESSURE':
            # Barometer always stays real
            sensors['baro_pressure'] = msg.press_abs
            sensors['baro_temp']     = msg.temperature

        if msg_type == 'GPS_RAW_INT':
            row = {'timestamp': round(elapsed, 3), 'label': label}
            row.update(sensors)
            writer.writerow(row)
            rows_written += 1

            if rows_written % 20 == 0:
                phase = "SPOOFING" if label == 1 else "normal"
                print(f"  {round(elapsed,1)}s [{phase}] — {rows_written} rows — GPS: {sensors['gps_lat']:.6f}, {sensors['gps_lon']:.6f}")

print(f"\nDone. {rows_written} rows written to {OUTPUT_FILE}")
