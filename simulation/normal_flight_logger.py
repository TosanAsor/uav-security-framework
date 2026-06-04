"""
normal_flight_logger.py
=======================
Logs real SITL sensor data for NORMAL flight (label=0).
Pure pymavlink — no DroneKit, Python 3.14 compatible.

USAGE:
  1. Launch SITL in terminal 1:
       pkill -f arducopter && pkill -f mavproxy && rm -f ~/ardupilot/mav.parm
       cd ~/ardupilot
       sim_vehicle.py -v ArduCopter -I0 --no-rebuild --wipe --speedup 10 \
         -l -35.363261,149.165230,584,353
     Wait for: "AP: EKF3 IMU0 is using GPS"

  2. In terminal 2, run this script:
       python3 normal_flight_logger.py --run 1
     Change --run 1 through 5 for each of the 5 runs.

OUTPUT: ../data/normal_flight_01.csv  (or _02 ... _05)
"""

import argparse
import csv
import math
import os
import time
from pymavlink import mavutil

# ── CONFIG ────────────────────────────────────────────────────────────────────
CONNECTION      = 'udp:127.0.0.1:14551'
OUTPUT_DIR      = '../data/'
LABEL           = 0
TAKEOFF_ALT     = 20       # metres above home
FLIGHT_SPEED    = 3        # m/s
FLIGHT_DURATION = 120      # seconds of logged flight after takeoff
LOG_HZ          = 10       # target log rate
LABEL_NAME      = 'normal_flight'

# Figure-8 waypoints (NED offsets in metres from home, repeated)
# vx, vy, duration_s
FLIGHT_PLAN = [
    ( FLIGHT_SPEED,  0,            10),
    ( 0,             FLIGHT_SPEED, 10),
    (-FLIGHT_SPEED,  0,            10),
    ( 0,            -FLIGHT_SPEED, 10),
    ( FLIGHT_SPEED,  0,            10),
    ( 0,             FLIGHT_SPEED, 10),
    (-FLIGHT_SPEED,  0,            10),
    ( 0,            -FLIGHT_SPEED, 10),
    ( 0,             0,            40),   # hover to fill 120s
]

FIELDS = [
    'timestamp',
    'gps_lat', 'gps_lon', 'gps_alt',
    'gps_fix_type', 'gps_satellites', 'gps_speed',
    'imu_xacc', 'imu_yacc', 'imu_zacc',
    'imu_xgyro', 'imu_ygyro', 'imu_zgyro',
    'baro_pressure', 'baro_temp',
    'mag_x', 'mag_y', 'mag_z',
    'label'
]

# ── HELPERS ───────────────────────────────────────────────────────────────────
def connect(connection_str):
    print(f"Connecting to {connection_str}...")
    master = mavutil.mavlink_connection(connection_str)
    master.wait_heartbeat()
    print(f"Connected. System {master.target_system} / Component {master.target_component}")
    # Request all data streams at 10 Hz
    master.mav.request_data_stream_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_ALL, LOG_HZ, 1
    )
    time.sleep(1)
    return master


def set_param(master, name, value):
    master.mav.param_set_send(
        master.target_system, master.target_component,
        name.encode(), float(value),
        mavutil.mavlink.MAV_PARAM_TYPE_REAL32
    )
    time.sleep(0.3)


def arm_and_takeoff(master, target_alt):
    print("Setting GUIDED mode...")
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        4, 0, 0, 0, 0, 0   # 4 = GUIDED for ArduCopter
    )
    time.sleep(1)

    print("Arming...")
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
        1, 0, 0, 0, 0, 0, 0
    )
    # Wait for ARMED confirmation
    for _ in range(30):
        msg = master.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
        if msg and (msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED):
            print("Armed.")
            break
        time.sleep(0.5)

    print(f"Taking off to {target_alt}m...")
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0,
        0, 0, 0, 0, 0, 0, target_alt
    )

    # Wait until altitude reached
    while True:
        msg = master.recv_match(type='GLOBAL_POSITION_INT',
                                blocking=True, timeout=2)
        if msg:
            alt = msg.relative_alt / 1000.0
            print(f"  Altitude: {alt:.1f}m", end='\r')
            if alt >= target_alt * 0.90:
                print(f"\nReached {alt:.1f}m — starting flight plan.")
                break
        time.sleep(0.2)


def send_velocity(master, vx, vy, vz=0):
    """Send NED velocity command in GUIDED mode."""
    master.mav.set_position_target_local_ned_send(
        0,
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b0000111111000111,   # velocity only
        0, 0, 0,
        vx, vy, vz,
        0, 0, 0,
        0, 0
    )


def collect_sensor_snapshot(master):
    """
    Non-blocking: read pending messages and update sensor dict.
    Returns the sensor dict (may have zeros if messages not yet received).
    """
    sensors = {
        'gps_lat': 0, 'gps_lon': 0, 'gps_alt': 0,
        'gps_fix_type': 0, 'gps_satellites': 0, 'gps_speed': 0,
        'imu_xacc': 0, 'imu_yacc': 0, 'imu_zacc': 0,
        'imu_xgyro': 0, 'imu_ygyro': 0, 'imu_zgyro': 0,
        'baro_pressure': 0, 'baro_temp': 0,
        'mag_x': 0, 'mag_y': 0, 'mag_z': 0,
    }
    return sensors


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=int, required=True,
                        help='Run number 1-5')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    outfile = os.path.join(OUTPUT_DIR, f'{LABEL_NAME}_{args.run:02d}.csv')
    print(f"Output: {outfile}")

    master = connect(CONNECTION)

    # Set params for clean flight
    set_param(master, 'DISARM_DELAY', 0)      # no auto-disarm
    set_param(master, 'SIM_SPEEDUP',  1)      # real-time logging
    set_param(master, 'FS_EKF_ACTION', 1)     # restore normal failsafe

    arm_and_takeoff(master, TAKEOFF_ALT)

    # ── SENSOR STATE (updated by message loop) ──
    sensors = {
        'gps_lat': 0, 'gps_lon': 0, 'gps_alt': 0,
        'gps_fix_type': 0, 'gps_satellites': 0, 'gps_speed': 0,
        'imu_xacc': 0, 'imu_yacc': 0, 'imu_zacc': 0,
        'imu_xgyro': 0, 'imu_ygyro': 0, 'imu_zgyro': 0,
        'baro_pressure': 0, 'baro_temp': 0,
        'mag_x': 0, 'mag_y': 0, 'mag_z': 0,
    }

    print(f"\nLogging {FLIGHT_DURATION}s of flight to {outfile}\n")
    start_time   = time.time()
    rows_written = 0
    plan_idx     = 0
    seg_end      = start_time + FLIGHT_PLAN[0][2]

    with open(outfile, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

        last_log = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed >= FLIGHT_DURATION:
                break

            # ── Update flight plan segment ──
            if time.time() >= seg_end and plan_idx < len(FLIGHT_PLAN) - 1:
                plan_idx += 1
                seg_end = time.time() + FLIGHT_PLAN[plan_idx][2]

            vx, vy, _ = FLIGHT_PLAN[plan_idx]
            send_velocity(master, vx, vy)

            # ── Read sensor messages (non-blocking) ──
            msg = master.recv_match(
                type=['GPS_RAW_INT', 'RAW_IMU', 'SCALED_PRESSURE'],
                blocking=False
            )
            if msg:
                t = msg.get_type()
                if t == 'GPS_RAW_INT':
                    sensors['gps_lat']        = msg.lat / 1e7
                    sensors['gps_lon']        = msg.lon / 1e7
                    sensors['gps_alt']        = msg.alt / 1000.0
                    sensors['gps_fix_type']   = msg.fix_type
                    sensors['gps_satellites'] = msg.satellites_visible
                    sensors['gps_speed']      = msg.vel / 100.0
                elif t == 'RAW_IMU':
                    sensors['imu_xacc']  = msg.xacc
                    sensors['imu_yacc']  = msg.yacc
                    sensors['imu_zacc']  = msg.zacc
                    sensors['imu_xgyro'] = msg.xgyro
                    sensors['imu_ygyro'] = msg.ygyro
                    sensors['imu_zgyro'] = msg.zgyro
                    sensors['mag_x']     = msg.xmag
                    sensors['mag_y']     = msg.ymag
                    sensors['mag_z']     = msg.zmag
                elif t == 'SCALED_PRESSURE':
                    sensors['baro_pressure'] = msg.press_abs
                    sensors['baro_temp']     = msg.temperature

            # ── Log at target Hz ──
            now = time.time()
            if now - last_log >= 1.0 / LOG_HZ:
                row = {'timestamp': round(elapsed, 3), 'label': LABEL}
                row.update(sensors)
                writer.writerow(row)
                rows_written += 1
                last_log = now

                if rows_written % 50 == 0:
                    print(f"  {elapsed:.1f}s — {rows_written} rows — "
                          f"GPS: {sensors['gps_lat']:.6f}, "
                          f"{sensors['gps_lon']:.6f} — "
                          f"alt: {sensors['gps_alt']:.1f}m")

    print(f"\nDone. {rows_written} rows written to {outfile}")
    master.close()


if __name__ == '__main__':
    main()
