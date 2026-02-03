import re
import serial
import time
from src.data.live.base import SensorDataSource
from src.data.models import SensorReading, IMUSample

G = 9.81  # m/s²

LINE_REGEX = re.compile(
    r"ACCEL \(g\):\s+"
    r"X=(?P<ax>-?\d+\.?\d*)\s+"
    r"Y=(?P<ay>-?\d+\.?\d*)\s+"
    r"Z=(?P<az>-?\d+\.?\d*).*?"
    r"Roll=(?P<roll>-?\d+\.?\d*)\s+"
    r"Pitch=(?P<pitch>-?\d+\.?\d*)"
)

# Alternative regex for Arduino output with separator
LINE_REGEX_ALT = re.compile(
    r"ACCEL \(g\):\s+"
    r"X=(?P<ax>-?\d+\.?\d*)\s+"
    r"Y=(?P<ay>-?\d+\.?\d*)\s+"
    r"Z=(?P<az>-?\d+\.?\d*).*?\|\s*"
    r"ANGLES \(deg\):\s+"
    r"Roll=(?P<roll>-?\d+\.?\d*)\s+"
    r"Pitch=(?P<pitch>-?\d+\.?\d*)"
)

class SerialIMUSensor(SensorDataSource):
    def __init__(self, port, baudrate=9600, timeout=1.0):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None

    def connect(self):
        self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
        time.sleep(2)                 # Arduino reset on macOS
        self.ser.reset_input_buffer()

    def disconnect(self):
        if self.ser:
            self.ser.close()
            self.ser = None

    def read(self, debug=False):
        line = self.ser.readline().decode(errors="ignore")

        if debug and line.strip():
            print(f"[RAW] {line.strip()}")

        # Try primary regex first
        match = LINE_REGEX.search(line)
        # Try alternative regex for Arduino output with separator
        if not match:
            match = LINE_REGEX_ALT.search(line)
        if not match:
            return None

        # Arduino outputs in g → convert to m/s²
        ax = float(match.group("ax")) * G
        ay = float(match.group("ay")) * G
        az = float(match.group("az")) * G

        # Create IMU sample (roll/pitch from Arduino but not stored in current model)
        imu = IMUSample(
            timestamp=time.time(),
            accel_x=ax,
            accel_y=ay,
            accel_z=az,
            gyro_x=0.0,  # Not provided by Arduino
            gyro_y=0.0,
            gyro_z=0.0,
        )

        return SensorReading(
            timestamp=time.time(),
            imu=imu,
            compression=None,
        )
