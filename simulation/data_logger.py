import dronekit
from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil
import csv
import time
import os
import threading
import math

# ── CONFIGURATION ──
OUTPUT_FILE  = '../data/normal_flight_01.csv'
LABEL        = 0
TAKEOFF_ALT  = 20
FLIGHT_SPEED = 3

# ── CONNECT ──
print("Connecting to SITL...")
vehicle = connect('tcp:127.0.0.1:5760', wait_ready=True, timeout=60)
print("Connected.")
print(f"  Firmware: {vehicle.version}")
print(f"  GPS fix: {vehicle.gps_0.fix_type}")
print(f"  Satellites: {vehicle.gps_0.satellites_visible}")

# ── HELPER FUNCTIONS ──
def arm_and_takeoff(alt):
    print("Waiting for vehicle to be armable...")
    while not vehicle.is_armable:
        print("  Not armable yet — waiting...")
        time.sleep(1)

    print("Arming...")
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    while not vehicle.armed:
        print("  Waiting for arm...")
        time.sleep(1)
    print("Armed.")

    print(f"Taking off to {alt}m...")
    vehicle.simple_takeoff(alt)

    while True:
        current_alt = vehicle.location.global_relative_frame.alt
        print(f"  Altitude: {current_alt:.1f}m", end='\r')
        if current_alt >= alt * 0.95:
            print(f"\nReached target altitude: {current_alt:.1f}m")
            break
        time.sleep(0.5)

def send_ned_velocity(vx, vy, vz):
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0, 0, 0,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b0000111111000111,
        0, 0, 0,
        vx, vy, vz,
        0, 0, 0,
        0, 0
    )
    vehicle.send_mavlink(msg)

def fly_segment(vx, vy, vz, duration):
    end = time.time() + duration
    while time.time() < end:
        send_ned_velocity(vx, vy, vz)
        time.sleep(0.1)

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

# ── FLIGHT PLAN ──
flight_plan = [
    (FLIGHT_SPEED,  0, 0, 10),
    (0,  FLIGHT_SPEED, 0, 10),
    (-FLIGHT_SPEED, 0, 0, 10),
    (0, -FLIGHT_SPEED, 0, 10),
    (FLIGHT_SPEED,  0, 0, 10),
    (0,  FLIGHT_SPEED, 0, 10),
    (-FLIGHT_SPEED, 0, 0, 10),
    (0, -FLIGHT_SPEED, 0, 10),
]

flight_done = threading.Event()

def fly():
    for vx, vy, vz, dur in flight_plan:
        print(f"\n  Flying: vx={vx} vy={vy} for {dur}s")
        fly_segment(vx, vy, vz, dur)
    # Hover
    print("\n  Hovering for 15s...")
    fly_segment(0, 0, 0, 15)
    flight_done.set()
    print("  Flight complete")

# ── ARM AND TAKE OFF ──
arm_and_takeoff(TAKEOFF_ALT)

# ── START LOGGING AND FLYING ──
print(f"\nLogging to {OUTPUT_FILE}\n")
start_time   = time.time()
rows_written = 0

flight_thread = threading.Thread(target=fly)
flight_thread.start()

with open(OUTPUT_FILE, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fields)
    writer.writeheader()

    while not flight_done.is_set():
        elapsed = time.time() - start_time

        loc     = vehicle.location.global_frame
        gps     = vehicle.gps_0
        att     = vehicle.attitude
        vel     = vehicle.velocity
        baro    = vehicle.location.global_relative_frame
        imu     = vehicle.raw_imu
        baro_p  = vehicle.scaled_pressure

        row = {
            'timestamp':      round(elapsed, 3),
            'gps_lat':        loc.lat if loc.lat else 0,
            'gps_lon':        loc.lon if loc.lon else 0,
            'gps_alt':        loc.alt if loc.alt else 0,
            'gps_fix_type':   gps.fix_type,
            'gps_satellites': gps.satellites_visible,
            'gps_speed':      math.sqrt(vel[0]**2 + vel[1]**2) if vel else 0,
            'imu_xacc':       imu.xacc if imu else 0,
            'imu_yacc':       imu.yacc if imu else 0,
            'imu_zacc':       imu.zacc if imu else 0,
            'imu_xgyro':      imu.xgyro if imu else 0,
            'imu_ygyro':      imu.ygyro if imu else 0,
            'imu_zgyro':      imu.zgyro if imu else 0,
            'baro_pressure':  baro_p.press_abs if baro_p else 0,
            'baro_temp':      baro_p.temperature if baro_p else 0,
            'mag_x':          imu.xmag if imu else 0,
            'mag_y':          imu.ymag if imu else 0,
            'mag_z':          imu.zmag if imu else 0,
            'label':          LABEL
        }

        writer.writerow(row)
        rows_written += 1

        if rows_written % 50 == 0:
            print(f"  {round(elapsed,1)}s — {rows_written} rows — "
                  f"GPS: {row['gps_lat']:.6f}, {row['gps_lon']:.6f} — "
                  f"alt: {baro.alt:.1f}m — "
                  f"speed: {row['gps_speed']:.1f}m/s")

        time.sleep(0.1)

flight_thread.join()
vehicle.close()
print(f"\nDone. {rows_written} rows written to {OUTPUT_FILE}")
