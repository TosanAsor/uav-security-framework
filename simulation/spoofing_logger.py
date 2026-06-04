"""
spoofing_logger.py
==================
Logs SITL sensor data with GPS SPOOFING injected at t=30s (label=1).
Attack injected via SIM_GPS1_GLTCH_X/Y parameters — the autopilot
genuinely sees and reacts to corrupted GPS. Pure pymavlink, no DroneKit.

Spoofing model:
  - t=0 to t=30s  : normal flight  (label=0, no glitch)
  - t=30s onward  : glitch grows gradually from 0 → GLITCH_MAX over
                    RAMP_DURATION seconds, then holds. Noise added each
                    step so early-onset rows overlap with normal — this
                    gives realistic (non-100%) spoofing recall.

USAGE:
  1. Launch SITL (terminal 1):
       pkill -f arducopter && pkill -f mavproxy && rm -f ~/ardupilot/mav.parm
       cd ~/ardupilot
       sim_vehicle.py -v ArduCopter -I0 --no-rebuild --wipe --speedup 10 \
         -l -35.363261,149.165230,584,353
     Wait for: "AP: EKF3 IMU0 is using GPS"

  2. Run script (terminal 2):
       python3 spoofing_logger.py --run 1
     Runs 1-5 use increasing GLITCH_MAX values (subtlest → strongest).

OUTPUT: ../data/gps_spoofing_01.csv  (or _02 ... _05)
"""

import argparse
import csv
import math
import os
import random
import time
from pymavlink import mavutil

# ── CONFIG ────────────────────────────────────────────────────────────────────
CONNECTION      = 'udp:127.0.0.1:14551'
OUTPUT_DIR      = '../data/'
TAKEOFF_ALT     = 20
FLIGHT_SPEED    = 3
FLIGHT_DURATION = 120
LOG_HZ          = 10
ATTACK_START    = 30       # seconds into flight

# Glitch magnitude per run (degrees) — subtlest to strongest
# These produce genuine difficulty variation across runs
GLITCH_MAX_PER_RUN = {
    1: 0.0008,
    2: 0.0012,
    3: 0.0016,
    4: 0.0020,
    5: 0.0025,
}

RAMP_DURATION = 20   # seconds to ramp from 0 to GLITCH_MAX
NOISE_STD     = 0.0001  # noise on glitch each timestep (creates early-onset ambiguity)

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
    print(f"Connected.")
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

def set_param_fast(master, name, value):
    """No-sleep param set for use inside the logging loop."""
    master.mav.param_set_send(
        master.target_system, master.target_component,
        name.encode(), float(value),
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


def glitch_magnitude(elapsed, attack_start, ramp_dur, glitch_max, noise_std):
    """
    Returns (gltch_x, gltch_y) for current timestep.
    Ramps from 0 → glitch_max over ramp_dur seconds after attack_start.
    Adds noise so early-onset rows overlap with normal flight.
    """
    if elapsed < attack_start:
        return 0.0, 0.0

    time_into_attack = elapsed - attack_start
    ramp_fraction    = min(time_into_attack / ramp_dur, 1.0)
    magnitude        = glitch_max * ramp_fraction

    # Add noise — this is what creates the early-onset ambiguity
    # that prevents trivial 100% recall
    noise_x = random.gauss(0, noise_std)
    noise_y = random.gauss(0, noise_std)

    # Split glitch across X and Y (45-degree direction)
    gltch_x = magnitude * 0.707 + noise_x
    gltch_y = magnitude * 0.707 + noise_y

    return gltch_x, gltch_y


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=int, required=True,
                        help='Run number 1-5')
    args = parser.parse_args()

    glitch_max = GLITCH_MAX_PER_RUN[args.run]
    outfile    = os.path.join(OUTPUT_DIR, f'gps_spoofing_{args.run:02d}.csv')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Run {args.run}: glitch_max={glitch_max} deg, "
          f"ramp over {RAMP_DURATION}s starting at t={ATTACK_START}s")
    print(f"Output: {outfile}")

    master = connect(CONNECTION)

    # Params for this run
    set_param(master, 'DISARM_DELAY',   0)
    set_param(master, 'SIM_SPEEDUP',    1)
    set_param(master, 'FS_EKF_ACTION',  0)   # report only — don't land during spoof
    set_param(master, 'FS_EKF_THRESH',  1.5)
    set_param(master, 'SIM_GPS1_GLTCH_X', 0) # ensure clean start
    set_param(master, 'SIM_GPS1_GLTCH_Y', 0)

    arm_and_takeoff(master, TAKEOFF_ALT)

    sensors = {
        'gps_lat': 0, 'gps_lon': 0, 'gps_alt': 0,
        'gps_fix_type': 0, 'gps_satellites': 0, 'gps_speed': 0,
        'imu_xacc': 0, 'imu_yacc': 0, 'imu_zacc': 0,
        'imu_xgyro': 0, 'imu_ygyro': 0, 'imu_zgyro': 0,
        'baro_pressure': 0, 'baro_temp': 0,
        'mag_x': 0, 'mag_y': 0, 'mag_z': 0,
    }

    print(f"\nLogging to {outfile}")
    print(f"Attack begins at t={ATTACK_START}s\n")

    start_time      = time.time()
    rows_written    = 0
    plan_idx        = 0
    seg_end         = start_time + FLIGHT_PLAN[0][2]
    last_log        = time.time()
    last_glitch_set = time.time()
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
            send_velocity(master, vx, vy)

            # ── Inject glitch every 0.5s (don't spam param_set) ──
            now = time.time()
            if now - last_glitch_set >= 0.5:
                gx, gy = glitch_magnitude(
                    elapsed, ATTACK_START, RAMP_DURATION,
                    glitch_max, NOISE_STD
                )
                set_param_fast(master, 'SIM_GPS1_GLTCH_X', gx)
                set_param_fast(master, 'SIM_GPS1_GLTCH_Y', gy)
                last_glitch_set = now
                current_label = 1 if elapsed >= ATTACK_START else 0

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
            if now - last_log >= 1.0 / LOG_HZ:
                row = {'timestamp': round(elapsed, 3), 'label': current_label}
                row.update(sensors)
                writer.writerow(row)
                rows_written += 1
                last_log = now

                if rows_written % 50 == 0:
                    phase = "SPOOFING" if current_label == 1 else "normal"
                    gx, gy = glitch_magnitude(
                        elapsed, ATTACK_START, RAMP_DURATION,
                        glitch_max, 0
                    )
                    print(f"  {elapsed:.1f}s [{phase}] — {rows_written} rows — "
                          f"glitch: ({gx:.5f}, {gy:.5f})")

    # Clean up — remove glitch
    set_param(master, 'SIM_GPS1_GLTCH_X', 0)
    set_param(master, 'SIM_GPS1_GLTCH_Y', 0)
    print(f"\nDone. {rows_written} rows written to {outfile}")
    master.close()


if __name__ == '__main__':
    main()
