"""
Physiological Data Source Abstraction Layer

Provides a unified interface for multiple data sources:
- Mock data generator (for testing)
- Live Arduino/Serial input (USB)
- BLE sensor input (placeholder)

All sources produce the same dict format expected by the dashboard.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import time
import numpy as np
from typing import Dict, Any, Optional, Protocol
from abc import ABC, abstractmethod
from enum import Enum

from src.data.models import (
    IMUSample, BloodPressureSample, CompressionSample,
    CompressionState, PostureState, SensorReading
)
from src.data.source import DataSource, MockDataSource


class DataSourceType(Enum):
    """Available data source types."""
    MOCK = "mock"
    SERIAL_ARDUINO = "serial_arduino"
    BLE = "ble"
    FILE = "file"


class IDataSource(Protocol):
    """
    Protocol for data sources.

    All data sources must implement this interface to be compatible
    with the dashboard.
    """

    def read(self) -> Dict[str, Any]:
        """
        Read a single sample from the data source.

        Returns:
            Dict with keys: timestamp, accel_x/y/z, gyro_x/y/z,
            posture (x/y/z dict), blood_pressure (dict with sbp/dbp/hr),
            compression (float), compression_state (str)
        """
        ...

    def close(self) -> None:
        """Close the data source and release resources."""
        ...

    def is_connected(self) -> bool:
        """Check if the data source is connected and operational."""
        ...


# =============================================================================
# Mock Data Source
# =============================================================================

class MockDataSourceAdapter:
    """
    Mock data source that generates simulated physiological data.

    Simulates realistic IMU, blood pressure, and compression data
    with periodic sit-to-stand events for testing.
    """

    SAMPLE_RATE = 50.0  # Hz
    GRAVITY = 9.81  # m/s²

    def __init__(
        self,
        sample_rate: float = SAMPLE_RATE,
        event_interval: float = 15.0,
    ):
        """
        Initialize mock data source.

        Args:
            sample_rate: Sampling frequency in Hz
            event_interval: Seconds between simulated stand events
        """
        self.sample_rate = sample_rate
        self.sample_interval = 1.0 / sample_rate
        self.event_interval = event_interval

        # Internal timing
        self._time = 0.0
        self._last_event_time = 0.0
        self._in_stand_event = False
        self._event_progress = 0.0

        # Base physiological values
        self._base_sbp = 120.0
        self._base_dbp = 80.0
        self._base_hr = 72.0

        # Compression simulation
        self._compression_pressure = 0.0
        self._compression_state = "Released"

    def read(self) -> Dict[str, Any]:
        """Generate a simulated sample."""
        self._time += self.sample_interval

        # Check if we should trigger a stand event
        time_since_event = self._time - self._last_event_time
        if time_since_event > self.event_interval and not self._in_stand_event:
            self._in_stand_event = True
            self._event_progress = 0.0

        # Simulate stand event (12 second duration to match state machine requirements)
        # State machine needs: 1s confirmation + 10s monitoring = 11s minimum
        posture_pitch = 45.0  # Seated pitch (degrees)
        accel_x = 0.0
        accel_y = 0.0
        accel_z = -self.GRAVITY

        if self._in_stand_event:
            self._event_progress += self.sample_interval

            # Sit-to-stand acceleration pattern
            if self._event_progress < 1.0:
                # Rising phase - upward acceleration spike with pitch transition
                progress = self._event_progress
                posture_pitch = 45.0 + 35.0 * progress  # Transition to standing

                # Add dynamic acceleration during rise
                rise_accel = 2.0 * np.sin(progress * np.pi)
                accel_z = -self.GRAVITY + rise_accel
                accel_y = 0.5 * np.sin(progress * 2 * np.pi)

            elif self._event_progress < 12.0:
                # Standing phase with BP drop (11 seconds to match state machine monitoring)
                posture_pitch = 80.0  # Standing (degrees)

                # Calculate accelerometer values from pitch angle
                # For abdomen sensor: pitch rotates gravity vector
                pitch_rad = np.radians(posture_pitch)
                accel_x = self.GRAVITY * np.sin(pitch_rad)
                accel_z = -self.GRAVITY * np.cos(pitch_rad)
                accel_y = 0.0

                # Add small postural sway during standing for detection
                sway = np.sin(self._event_progress * 2.0) * 0.15
                accel_z += sway
                accel_y = sway * 0.5

            else:
                # Event complete - return to seated
                self._in_stand_event = False
                self._last_event_time = self._time
                posture_pitch = 45.0  # Seated

        else:
            # Not in stand event - calculate accel from seated pitch
            pitch_rad = np.radians(posture_pitch)
            accel_x = self.GRAVITY * np.sin(pitch_rad)
            accel_z = -self.GRAVITY * np.cos(pitch_rad)
            accel_y = 0.0

        # Add noise to IMU
        accel_x += np.random.normal(0, 0.05)
        accel_y += np.random.normal(0, 0.03)
        accel_z += np.random.normal(0, 0.02)

        # Simulate BP response to standing
        sbp_drop = 0.0
        if self._in_stand_event and self._event_progress > 1.0:
            # BP drops during standing (over 5 seconds, then recovers)
            drop_progress = min(1.0, (self._event_progress - 1.0) / 5.0)
            sbp_drop = 20.0 * drop_progress

            # Start recovery after peak drop (at 6 seconds)
            if self._event_progress > 6.0:
                recovery_progress = min(1.0, (self._event_progress - 6.0) / 5.0)
                sbp_drop = 20.0 * (1.0 - 0.7 * recovery_progress)  # Recover to 30% of max drop

        sbp = self._base_sbp - sbp_drop + np.random.normal(0, 2.0)
        dbp = self._base_dbp - sbp_drop * 0.6 + np.random.normal(0, 1.5)
        hr = self._base_hr + sbp_drop * 0.3 + np.random.normal(0, 2.0)

        # Simulate compression
        if self._compression_state == "Engaging":
            self._compression_pressure = min(20.0, self._compression_pressure + 5.0 * self.sample_interval)
            if self._compression_pressure >= 19.0:
                self._compression_state = "Holding"
        elif self._compression_state == "Holding":
            # Hold for a bit then release
            if np.random.random() < 0.01:  # Random chance to start releasing
                self._compression_state = "Releasing"
        elif self._compression_state == "Releasing":
            self._compression_pressure = max(0.0, self._compression_pressure - 5.0 * self.sample_interval)
            if self._compression_pressure <= 0.1:
                self._compression_state = "Released"

        return {
            'timestamp': self._time,
            'accel_x': float(accel_x),
            'accel_y': float(accel_y),
            'accel_z': float(accel_z),
            'gyro_x': np.random.normal(0, 0.01),
            'gyro_y': np.random.normal(0, 0.01),
            'gyro_z': np.random.normal(0, 0.01),
            'posture': {
                'x': float(accel_x),
                'y': float(accel_y),
                'z': float(accel_z)
            },
            'blood_pressure': {
                'systolic': float(sbp),
                'diastolic': float(dbp),
                'hr': float(hr)
            },
            'compression': self._compression_pressure,
            'compression_state': self._compression_state
        }

    def close(self) -> None:
        """No resources to release."""
        pass

    def is_connected(self) -> bool:
        """Mock source is always connected."""
        return True

    def get_status(self) -> Dict[str, Any]:
        """Get status for mock data source."""
        return {
            'connected': True,
            'port': 'mock',
            'baudrate': 0,
            'successful_reads': self._time * self.sample_rate,
            'errors': 0,
            'success_rate': 1.0
        }


# =============================================================================
# Serial/Arduino Data Source
# =============================================================================

class SerialDataSourceAdapter:
    """
    Live data source from Arduino via serial/USB connection.

    Reads IMU data from Arduino and generates simulated BP data.
    """

    def __init__(
        self,
        port: str = "/dev/tty.usbserial-DN04ABAX",
        baudrate: int = 9600,
        sample_rate: float = 50.0,
    ):
        """
        Initialize serial data source.

        Args:
            port: Serial port path (platform-dependent)
            baudrate: Serial communication baud rate
            sample_rate: Target sampling rate in Hz
        """
        self.port = port
        self.baudrate = baudrate
        self.sample_rate = sample_rate
        self.sample_interval = 1.0 / sample_rate

        self._sensor = None
        self._connected = False
        self._time = 0.0
        self._read_count = 0  # Track successful Arduino reads
        self._error_count = 0  # Track read errors

        # BP simulation (Arduino doesn't provide BP)
        self._base_bp_sys = 120.0
        self._base_bp_dia = 80.0
        self._base_hr = 72.0
        self._last_bp_time = -1.0
        self._bp_interval = 1.0  # Generate BP every second

        # Connect to sensor
        self._connect()

    def _connect(self) -> bool:
        """Attempt to connect to Arduino sensor."""
        try:
            from src.data.live.serial_source import SerialIMUSensor
            self._sensor = SerialIMUSensor(port=self.port, baudrate=self.baudrate)
            self._sensor.connect()
            self._connected = True
            print(f"✓ Connected to Arduino on {self.port}")
            return True
        except Exception as e:
            print(f"✗ Could not connect to Arduino on {self.port}: {e}")
            self._connected = False
            return False

    def read(self) -> Dict[str, Any]:
        """
        Read from Arduino and generate sample.

        Returns dict with simulated BP if Arduino not available.
        """
        self._time += self.sample_interval

        # Default values
        result = {
            'timestamp': self._time,
            'accel_x': 0.0,
            'accel_y': 0.0,
            'accel_z': -9.81,
            'gyro_x': 0.0,
            'gyro_y': 0.0,
            'gyro_z': 0.0,
            'posture': {'x': 0.0, 'y': 0.0, 'z': -9.81},
            'blood_pressure': {'systolic': 120.0, 'diastolic': 80.0, 'hr': 72.0},
            'compression': 0.0,
            'compression_state': 'Released',
            'data_source': 'mock'  # Track data source
        }

        # Try to read from Arduino
        if self._connected and self._sensor:
            try:
                arduino_reading = self._sensor.read()
                if arduino_reading is not None and arduino_reading.imu:
                    imu = arduino_reading.imu
                    self._read_count += 1
                    result.update({
                        'accel_x': imu.accel_x,
                        'accel_y': imu.accel_y,
                        'accel_z': imu.accel_z,
                        'gyro_x': 0.0,  # Arduino provides orientation, not gyro
                        'gyro_y': 0.0,
                        'gyro_z': 0.0,
                        'posture': {
                            'x': imu.accel_x,
                            'y': imu.accel_y,
                            'z': imu.accel_z
                        },
                        'data_source': 'arduino'
                    })
                else:
                    self._error_count += 1
            except Exception as e:
                self._error_count += 1
                print(f"Error reading from Arduino: {e}")
                # Try to reconnect
                self._connected = False
                self._connect()
        else:
            self._error_count += 1

        # Generate BP data (simulated)
        if self._time - self._last_bp_time >= self._bp_interval:
            import random
            import math

            # Add variability
            sys_var = random.gauss(0, 5.0)
            dia_var = random.gauss(0, 3.0)
            hr_var = random.gauss(0, 3.0)

            # Respiration-induced oscillation
            resp_osc = 0.1 * math.sin(2 * math.pi * 0.2 * self._time)

            result['blood_pressure'] = {
                'systolic': self._base_bp_sys + sys_var + resp_osc * 5,
                'diastolic': self._base_bp_dia + dia_var + resp_osc * 3,
                'hr': self._base_hr + hr_var + resp_osc * 2
            }
            self._last_bp_time = self._time

        return result

    def close(self) -> None:
        """Close Arduino connection."""
        if self._sensor and hasattr(self._sensor, 'disconnect'):
            self._sensor.disconnect()
        self._connected = False

    def is_connected(self) -> bool:
        """Check if Arduino is connected."""
        return self._connected

    def get_status(self) -> Dict[str, Any]:
        """Get connection status and statistics."""
        return {
            'connected': self._connected,
            'port': self.port,
            'baudrate': self.baudrate,
            'successful_reads': self._read_count,
            'errors': self._error_count,
            'success_rate': self._read_count / max(1, self._read_count + self._error_count)
        }


# =============================================================================
# BLE Data Source (Placeholder)
# =============================================================================

class BLEDataSourceAdapter:
    """
    Placeholder for BLE sensor data source.

    Implements the interface but returns mock data with a warning.
    Can be extended to connect to actual BLE devices.

    Example BLE implementation using bleak library:

    ```python
    from bleak import BleakClient

    class RealBLEDataSourceAdapter(BLEDataSourceAdapter):
        async def connect_async(self) -> bool:
            async with BleakClient(self.device_address) as client:
                await client.connect()
                # Subscribe to IMU characteristic
                await client.start_notify(IMU_UUID, self._notification_handler)
                self._connected = True
                return True

        def _notification_handler(self, sender, data):
            # Parse BLE data packet
            self._parse_imu_data(data)
    ```
    """

    def __init__(
        self,
        device_address: Optional[str] = None,
        sample_rate: float = 50.0,
    ):
        """
        Initialize BLE data source.

        Args:
            device_address: BLE MAC address or UUID (platform-specific)
            sample_rate: Target sampling rate in Hz
        """
        self.device_address = device_address or "00:00:00:00:00:00"
        self.sample_rate = sample_rate
        self._time = 0.0
        self._connected = False
        self._mock = MockDataSourceAdapter(sample_rate=sample_rate)

        print(f"BLE data source initialized for {self.device_address}")
        print("Note: This is a placeholder. Install 'bleak' and implement")
        print("async connection for production BLE support.")

    def read(self) -> Dict[str, Any]:
        """Read sample (returns mock data)."""
        if not self._connected:
            # Fall back to mock data
            sample = self._mock.read()
        else:
            # Would read from BLE characteristic
            sample = self._mock.read()

        return sample

    def close(self) -> None:
        """Close BLE connection."""
        self._connected = False
        self._mock.close()

    def is_connected(self) -> bool:
        """Check if BLE is connected."""
        return self._connected


# =============================================================================
# Unified Data Source Factory
# =============================================================================

class PhysiologicalDataSource:
    """
    Unified data source that supports multiple input types.

    Usage:
        # Use mock data (default)
        source = PhysiologicalDataSource()

        # Use Arduino/Serial input
        source = PhysiologicalDataSource(source_type=DataSourceType.SERIAL_ARDUINO)

        # Use BLE input
        source = PhysiologicalDataSource(source_type=DataSourceType.BLE)

        # Read samples
        while True:
            sample = source.read()
            process(sample)
    """

    def __init__(
        self,
        source_type: DataSourceType = DataSourceType.MOCK,
        port: Optional[str] = None,
        device_address: Optional[str] = None,
        sample_rate: float = 50.0,
        event_interval: float = 15.0,
    ):
        """
        Initialize unified data source.

        Args:
            source_type: Type of data source to use
            port: Serial port (for SERIAL_ARDUINO)
            device_address: BLE device address (for BLE)
            sample_rate: Sampling rate in Hz
            event_interval: Seconds between events (for MOCK)
        """
        self.source_type = source_type
        self.sample_rate = sample_rate

        # Create appropriate adapter
        if source_type == DataSourceType.MOCK:
            self._adapter = MockDataSourceAdapter(
                sample_rate=sample_rate,
                event_interval=event_interval,
            )
        elif source_type == DataSourceType.SERIAL_ARDUINO:
            self._adapter = SerialDataSourceAdapter(
                port=port or self._detect_serial_port(),
                sample_rate=sample_rate,
            )
        elif source_type == DataSourceType.BLE:
            self._adapter = BLEDataSourceAdapter(
                device_address=device_address,
                sample_rate=sample_rate,
            )
        else:
            print(f"Unknown source type: {source_type}, using MOCK")
            self._adapter = MockDataSourceAdapter(sample_rate=sample_rate)

    def _detect_serial_port(self) -> str:
        """
        Attempt to detect the correct serial port.

        Returns a platform-appropriate default port.
        """
        import platform
        system = platform.system()

        if system == "Darwin":  # macOS
            return "/dev/tty.usbserial-DN04ABAX"
        elif system == "Linux":
            return "/dev/ttyUSB0"
        elif system == "Windows":
            return "COM3"
        else:
            return "/dev/ttyUSB0"

    def read(self) -> Dict[str, Any]:
        """
        Read a sample from the configured data source.

        Returns:
            Dict with all sensor data for dashboard processing
        """
        return self._adapter.read()

    def close(self) -> None:
        """Close the data source."""
        self._adapter.close()

    def is_connected(self) -> bool:
        """Check if data source is connected."""
        return self._adapter.is_connected()

    def switch_source(self, new_source_type: DataSourceType, **kwargs) -> None:
        """
        Switch to a different data source at runtime.

        Args:
            new_source_type: Type of source to switch to
            **kwargs: Additional arguments for the new source
        """
        self.close()
        self.source_type = new_source_type

        if new_source_type == DataSourceType.MOCK:
            self._adapter = MockDataSourceAdapter(
                sample_rate=self.sample_rate,
                event_interval=kwargs.get('event_interval', 15.0),
            )
        elif new_source_type == DataSourceType.SERIAL_ARDUINO:
            self._adapter = SerialDataSourceAdapter(
                port=kwargs.get('port', self._detect_serial_port()),
                sample_rate=self.sample_rate,
            )
        elif new_source_type == DataSourceType.BLE:
            self._adapter = BLEDataSourceAdapter(
                device_address=kwargs.get('device_address'),
                sample_rate=self.sample_rate,
            )


# =============================================================================
# Legacy Compatibility (for existing dashboard code)
# =============================================================================

class PhysiologicalDataGenerator(PhysiologicalDataSource):
    """
    Legacy alias for PhysiologicalDataSource.

    Maintains compatibility with existing dashboard code while
    supporting the new abstracted interface.
    """

    def __init__(
        self,
        data_source: Optional[DataSource] = None,
        sample_rate: float = 50.0,
        event_interval: float = 15.0,
        use_mock: bool = True,
        serial_port: Optional[str] = None,
    ):
        """
        Initialize with legacy parameters for backward compatibility.

        Args:
            data_source: Optional existing DataSource (ignored, using new interface)
            sample_rate: Sampling frequency in Hz
            event_interval: Seconds between simulated stand events
            use_mock: If True, use mock data; if False, try serial
            serial_port: Serial port for live data
        """
        # Determine source type from legacy parameters
        if use_mock:
            source_type = DataSourceType.MOCK
        else:
            source_type = DataSourceType.SERIAL_ARDUINO

        # Initialize parent with new interface
        super().__init__(
            source_type=source_type,
            port=serial_port,
            sample_rate=sample_rate,
            event_interval=event_interval,
        )

    def generate_sample(self) -> Dict[str, Any]:
        """
        Legacy method name for read().

        Maintains compatibility with existing dashboard code.
        """
        return self.read()


# Export
__all__ = [
    'DataSourceType',
    'PhysiologicalDataSource',
    'PhysiologicalDataGenerator',
    'MockDataSourceAdapter',
    'SerialDataSourceAdapter',
    'BLEDataSourceAdapter',
]
