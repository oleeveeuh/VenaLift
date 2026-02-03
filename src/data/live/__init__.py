"""
Live sensor connection modules.

TODO: Implement these modules to connect to real hardware.

Each module should inherit from DataSource and implement:
- read() -> SensorReading: Return synchronized sensor data
- close(): Clean up connections

Example implementations are provided as templates.
"""

# TODO: Uncomment when implementing live sensors
from .imu_sensor import LiveIMUSensor
from .bp_sensor import LiveBPSensor
from .compression_device import LiveCompressionDevice

__all__ = [
    "LiveIMUSensor",
    "LiveBPSensor",
    "LiveCompressionDevice",
]
