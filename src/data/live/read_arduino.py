# data/live/read_arduino.py

import time
import sys
from .serial_source import SerialIMUSensor


def main():
    debug = "--debug" in sys.argv

    sensor = SerialIMUSensor(
        port="/dev/tty.usbserial-DN04ABAX",  # CHANGE THIS
        baudrate=9600
    )

    sensor.connect()

    print("Reading from Arduino...")
    print("=" * 70)
    print(f"{'Time':<10} {'Accel X':<12} {'Accel Y':<12} {'Accel Z':<12} {'Magnitude':<12}")
    print("-" * 70)

    try:
        start_time = time.time()
        count = 0

        while True:
            reading = sensor.read(debug=debug)
            if reading is not None:
                elapsed = time.time() - start_time
                ax = reading.imu.accel_x
                ay = reading.imu.accel_y
                az = reading.imu.accel_z
                mag = (ax**2 + ay**2 + az**2)**0.5

                print(f"{elapsed:8.1f}s  {ax:+10.3f}  {ay:+10.3f}  {az:+10.3f}  {mag:10.3f}")

                count += 1
                if count % 10 == 0:
                    print("-" * 70)
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        sensor.close()


if __name__ == "__main__":
    main()
