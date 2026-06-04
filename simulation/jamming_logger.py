"""
jamming_logger.py  (PULSED VERSION)
===================================
Logs SITL sensor data with PULSED GPS JAMMING (label=2).

Real SITL jamming (SIM_GPS1_JAM=1) corrupts GPS accuracy/variance and
crashes the drone ~10s after onset. To capture genuine jammed-FLIGHT
data without the crash, this logger PULSES the jam:

    jam ON  for JAM_ON_SEC  -> drone experiences GPS loss (label=2)
    jam OFF for JAM_OFF_SEC -> drone recovers position    (label=0)
    repeat

This gives multiple clean airborne jam episodes per flight and avoids
the "crashed on ground" contamination. Labels: 2 during each jam pulse,
0 otherwise.

NOTE: The jam signature is NOT in gps_fix_type/satellites (those stay
6/10 in this ArduPilot version). The real signal is in the EKF residuals
(position errors / variance) which ekf_processor.py computes downstream.

USAGE:
  1. Launch SITL (terminal 1):
       pkill -9 -f arducopter; pkill -9 -f mavproxy; pkill -9 -f sim_vehicle; sleep 3
       cd ~/ardupilot
       sim_vehicle.py -v ArduCopter -I0 --no-rebuild --speedup 10 \
         -l -35.363261,149.165230,584,353 --out udp:127.0.0.1:14551
     Wait for "AP: EKF3 IMU0 is using GPS", then: param set DISARM_DELAY 0

  2. Run script (terminal 2):
       python3 jamming_logger.py --run 1
     Runs 1-5 vary the first-jam timing slightly for diversity.

OUTPUT: ../data/gps_jamming_01.csv  (or _02 ... _05)
"""

import argparse
import csv
import os
import time
from pymavlink import mavutil

# ── CONFIG ────────────────────────────────────────────────────────────────────
CONNECTION      = 'udp:127.0.0.1:14551'
OUTPUT_DIR      = '../data/'
TAKEOFF_ALT     = 20
FLIGHT_SPEED    = 3
FLIGHT_DURATION = 120
LOG_HZ          = 10

# Pulsed jamming schedule
JAM_ON_SEC      = 8     # jam active duration (short enough to avoid crash)
JAM_OFF_SEC     = 10    # recovery duration between pulses
FIRST_JAM_PER_RUN = {1: 25, 2: 27, 3: 30, 4: 33, 5: 35}  # first jam onset

FLIGHT_PLAN = [
    ( FLIGHT_SPEED,  0,            10),
    ( 0,             FLIGHT_SPEED, 10),
    (-FLIGHT_SPEED,  0,            10),
    ( 0,            -FLIGHT_SPEED, 10),
    ( FLIGHT_SPEED,  0,            10),
    ( 0,             FLIGHT_SPEED, 10),
    (-FLIGHT_SPEED,  0,            10),
    ( 0,            -FLIGHT_SPEED, 10),
    ( 0,             0,            40),
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
    print("Connected.")
    master.mav.request_data_stream_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_ALL, LOG_HZ, 1
    )
    time.sleep(1)
    return master


def set_param(master, name, value):
    """Blocking param set used during setup."""
    master.mav.param_set_send(
        master.target_system, master.target_component,
        name.encode(), float(value),
        mavutil.mavlink.MAV_PARAM_TYPE_REAL32
    )
    time.sleep(0.3)


def set_jam(master, on):
    """Toggle jam. Called only at pulse edges, so no sleep needed."""
    master.mav.param_set_send(
        master.target_system, master.target_component,
        b'SIM_GPS1_JAM', float(1 if on else 0),
        mavutil.mavlink.MAV_PARAM_TYPE_REAL32
    )


def arm_and_takeoff(master, target_alt):
    print("Setting GUIDED mode...")
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        4, 0, 0, 0, 0, 0
    )
    time.sleep(1)

    print("Arming...")
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
        1, 0, 0, 0, 0, 0, 0
    )
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
    while True:
        msg = master.recv_match(type='GLOBAL_POSITION_INT',
                                blocking=True, timeout=2)
        if msg:
            alt = msg.relative_alt / 1000.0
            print(f"  Altitude: {alt:.1f}m", end='\r')
            if alt >= target_alt * 0.90:
                print(f"\nReached {alt:.1f}m — starting flight.")
                break
        time.sleep(0.2)


def send_velocity(master, vx, vy, vz=0):
    master.mav.set_position_target_local_ned_send(
        0, master.target_system, master.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b0000111111000111,
        0, 0, 0, vx, vy, vz, 0, 0, 0, 0, 0
    )


def reestablish_guided(master):
    """After a jam pulse forces ALT_HOLD failsafe, switch back to GUIDED."""
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        4, 0, 0, 0, 0, 0
    )


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=int, required=True, help='Run number 1-5')
    args = parser.parse_args()

    first_jam = FIRST_JAM_PER_RUN[args.run]
    outfile   = os.path.join(OUTPUT_DIR, f'gps_jamming_{args.run:02d}.csv')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Run {args.run}: first jam at t={first_jam}s, "
          f"pulse {JAM_ON_SEC}s ON / {JAM_OFF_SEC}s OFF")
    print(f"Output: {outfile}")

    master = connect(CONNECTION)

    set_param(master, 'DISARM_DELAY',  0)
    set_param(master, 'SIM_SPEEDUP',   1)
    set_param(master, 'FS_EKF_ACTION', 2)   # AltHold on failsafe (not land)
    set_param(master, 'FS_EKF_THRESH', 0.8)
    set_param(master, 'SIM_GPS1_JAM',  0)   # clean start

    arm_and_takeoff(master, TAKEOFF_ALT)

    sensors = {
        'gps_lat': 0, 'gps_lon': 0, 'gps_alt': 0,
        'gps_fix_type': 0, 'gps_satellites': 0, 'gps_speed': 0,
        'imu_xacc': 0, 'imu_yacc': 0, 'imu_zacc': 0,
        'imu_xgyro': 0, 'imu_ygyro': 0, 'imu_zgyro': 0,
        'baro_pressure': 0, 'baro_temp': 0,
        'mag_x': 0, 'mag_y': 0, 'mag_z': 0,
    }

    print(f"\nLogging to {outfile}\n")

    start_time      = time.time()
    rows_written    = 0
    plan_idx        = 0
    seg_end         = start_time + FLIGHT_PLAN[0][2]
    last_log        = time.time()

    jam_active      = False
    next_jam_toggle = first_jam     # time of next on/off switch
    current_label   = 0

    with open(outfile, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

        while True:
            elapsed = time.time() - start_time
            if elapsed >= FLIGHT_DURATION:
                break

            # ── Flight plan ──
            if time.time() >= seg_end and plan_idx < len(FLIGHT_PLAN) - 1:
                plan_idx += 1
                seg_end = time.time() + FLIGHT_PLAN[plan_idx][2]
            vx, vy, _ = FLIGHT_PLAN[plan_idx]
            # only command velocity when not jammed (no position control while jammed)
            if not jam_active:
                send_velocity(master, vx, vy)

            # ── Pulse the jam ──
            if elapsed >= next_jam_toggle:
                if not jam_active:
                    set_jam(master, True)
                    jam_active      = True
                    current_label   = 2
                    next_jam_toggle = elapsed + JAM_ON_SEC
                    print(f"  >> JAM ON  at t={elapsed:.1f}s")
                else:
                    set_jam(master, False)
                    reestablish_guided(master)
                    jam_active      = False
                    current_label   = 0
                    next_jam_toggle = elapsed + JAM_OFF_SEC
                    print(f"  >> JAM OFF at t={elapsed:.1f}s (recovering)")

            # ── Read sensors ──
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
                row = {'timestamp': round(elapsed, 3), 'label': current_label}
                row.update(sensors)
                writer.writerow(row)
                rows_written += 1
                last_log = now

                if rows_written % 50 == 0:
                    phase = "JAMMING" if current_label == 2 else "normal"
                    print(f"  {elapsed:.1f}s [{phase}] — {rows_written} rows — "
                          f"alt: {sensors['gps_alt']:.1f}m")

    # Clean up
    set_jam(master, False)
    print(f"\nDone. {rows_written} rows written to {outfile}")
    master.close()


if __name__ == '__main__':
    main()
