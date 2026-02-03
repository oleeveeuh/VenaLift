"""
Data source abstraction and mock implementation.

Defines the interface for sensor data sources and provides a mock implementation
for testing and development. Live sensor implementations should inherit from
the base DataSource class.
"""

from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime
import math
import random
import time
from typing import Generator, Optional

import numpy as np

from .models import (
    IMUSample,
    BloodPressureSample,
    CompressionSample,
    CompressionState,
    SensorReading,
)


class DataSource(ABC):
    """
    Abstract base class for sensor data sources.

    All sensor data sources (mock or live) should inherit from this class
    and implement the read() method.
    """

    @abstractmethod
    def read(self) -> SensorReading:
        """
        Read a single synchronized sensor reading.

        Returns:
            SensorReading with current timestamp and available sensor data.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Clean up resources (close connections, stop threads, etc.)."""
        pass


class MockDataSource(DataSource):
    """
    Mock data source for testing and development.

    Simulates realistic physiological signals:
    - IMU: posture-dependent acceleration with noise
    - Blood pressure: periodic readings with variability
    - Compression: simple state machine simulation

    Generates realistic sit-to-stand events for testing detection.
    """

    # Simulation parameters
    SAMPLE_RATE = 50.0  # Hz
    GRAVITY = 9.81  # m/s²

    # Posture acceleration patterns (approximate average values)
    # Z-axis acceleration in different postures (negative = upward on chest)
    POSTURE_ACCEL = {
        "lying": (0, 0, -9.81),  # Flat on back
        "sitting": (0, 0, -9.81),  # Upright
        "standing": (0, 0, -9.81),  # Upright
    }

    def __init__(
        self,
        sample_rate: float = SAMPLE_RATE,
        event_interval: float = 30.0,  # Seconds between stand events
        noise_level: float = 0.1,  # Accelerometer noise std dev (m/s²)
        bp_interval: float = 1.0,  # Seconds between BP readings
    ):
        """
        Initialize mock data source.

        Args:
            sample_rate: Sampling frequency in Hz
            event_interval: Average seconds between simulated stand events
            noise_level: Standard deviation of accelerometer noise
            bp_interval: Seconds between blood pressure readings
        """
        self.sample_rate = sample_rate
        self.sample_interval = 1.0 / sample_rate
        self.event_interval = event_interval
        self.noise_level = noise_level
        self.bp_interval = bp_interval

        # State tracking
        self._time = 0.0
        self._posture = "sitting"
        self._last_bp_time = -bp_interval - 1
        self._next_event_time = event_interval

        # Compression simulation
        self._compression_pressure = 0.0
        self._compression_state = CompressionState.OFF

        # Base physiological values
        self._base_bp_sys = 120.0
        self._base_bp_dia = 80.0
        self._base_hr = 72.0

    def read(self) -> SensorReading:
        """Generate the next mock sensor reading."""
        # Update time
        self._time += self.sample_interval

        # Check if we should trigger a stand event
        if self._time >= self._next_event_time:
            self._trigger_stand_event()
            # Schedule next event with some randomness
            self._next_event_time = self._time + self.event_interval * (0.8 + 0.4 * random.random())

        # Generate IMU data
        imu = self._generate_imu()

        # Generate blood pressure data periodically
        bp = self._generate_bp() if self._time - self._last_bp_time >= self.bp_interval else None
        if bp is not None:
            self._last_bp_time = self._time

        # Generate compression data
        compression = self._generate_compression()

        return SensorReading(
            timestamp=self._time,
            imu=imu,
            blood_pressure=bp,
            compression=compression,
        )

    def _trigger_stand_event(self) -> None:
        """Transition from sitting to standing posture."""
        self._posture = "standing"
        # Schedule return to sitting after 2-5 seconds
        self._return_time = self._time + 2.0 + 3.0 * random.random()

    def _generate_imu(self) -> IMUSample:
        """Generate IMU sample with current posture and noise."""
        # Get base acceleration for current posture
        base_accel = self.POSTURE_ACCEL.get(self._posture, (0, 0, -self.GRAVITY))

        # Add noise
        noise = np.random.normal(0, self.noise_level, 3)
        accel = np.array(base_accel) + noise

        # Add small random rotation (gyro)
        gyro = np.random.normal(0, 0.01, 3)  # rad/s

        # Add transient acceleration during posture transitions
        if self._posture == "standing" and hasattr(self, "_return_time"):
            time_since_stand = self._time - (self._next_event_time - self.event_interval)
            if time_since_stand < 1.0:  # First second of stand
                # Add upward acceleration spike characteristic of sit-to-stand
                spike = math.exp(-10 * (time_since_stand - 0.3) ** 2)  # Gaussian spike
                accel[2] -= spike * 2.0  # Upward acceleration (negative on chest-mounted IMU)
                # Add some forward-backward motion
                accel[0] = spike * 0.5
                # Add rotation during stand
                gyro[1] = spike * 0.5

            # Return to sitting
            if self._time >= self._return_time:
                self._posture = "sitting"

        return IMUSample(
            timestamp=self._time,
            accel_x=float(accel[0]),
            accel_y=float(accel[1]),
            accel_z=float(accel[2]),
            gyro_x=float(gyro[0]),
            gyro_y=float(gyro[1]),
            gyro_z=float(gyro[2]),
        )

    def _generate_bp(self) -> Optional[BloodPressureSample]:
        """Generate blood pressure sample with physiological variability."""
        # Add natural variability
        sys_var = random.gauss(0, 5.0)
        dia_var = random.gauss(0, 3.0)
        hr_var = random.gauss(0, 3.0)

        # Respiration-induced oscillation (~0.2 Hz)
        resp_osc = 0.1 * math.sin(2 * math.pi * 0.2 * self._time)

        # Stress response during standing
        if self._posture == "standing" and hasattr(self, "_return_time"):
            time_since_stand = self._time - (self._next_event_time - self.event_interval)
            if time_since_stand < 2.0:
                # Brief BP/HR increase on standing
                stress = math.exp(-time_since_stand) * 10.0
                sys_var += stress
                hr_var += stress * 0.5

        return BloodPressureSample(
            timestamp=self._time,
            systolic=self._base_bp_sys + sys_var + resp_osc * 5,
            diastolic=self._base_bp_dia + dia_var + resp_osc * 3,
            heart_rate=self._base_hr + hr_var + resp_osc * 2,
        )

    def _generate_compression(self) -> Optional[CompressionSample]:
        """Generate compression device reading (simplified simulation)."""
        # Very basic state machine - just cycles when triggered
        # In real implementation, this would interface with actual device
        return None  # No active compression in basic simulation

    def close(self) -> None:
        """Clean up (no-op for mock source)."""
        pass


class LiveDataSource(DataSource):
    """
    Template for live sensor data source implementation.

    To implement live data collection:
    1. Subclass this class
    2. Initialize sensor connections in __init__
    3. Implement read() to return synchronized SensorReading
    4. Implement close() to properly shut down connections

    Example:
        class MyLiveSource(LiveDataSource):
            def __init__(self, imu_port: str, bp_port: str):
                self.imu = IMUSensor(port=imu_port)
                self.bp = BPSensor(port=bp_port)

            def read(self) -> SensorReading:
                t = time.time()
                return SensorReading(
                    timestamp=t,
                    imu=self.imu.read(),
                    blood_pressure=self.bp.read(),
                )
    """

    def __init__(self):
        """Initialize live sensor connections."""
        # TODO: Initialize actual sensor connections
        raise NotImplementedError("Implement live sensor connections")

    def read(self) -> SensorReading:
        """Read from live sensors."""
        raise NotImplementedError("Implement live sensor reading")

    def close(self) -> None:
        """Close sensor connections."""
        raise NotImplementedError("Implement connection cleanup")


class DataBuffer:
    """
    Circular buffer for real-time sensor data.

    Maintains a sliding window of recent sensor readings for feature extraction
    and event detection.
    """

    def __init__(self, duration: float, sample_rate: float):
        """
        Initialize buffer.

        Args:
            duration: Buffer duration in seconds
            sample_rate: Sampling rate in Hz
        """
        self.capacity = int(duration * sample_rate)
        self._buffer: deque[SensorReading] = deque(maxlen=self.capacity)

    @property
    def size(self) -> int:
        """Current number of samples in buffer."""
        return len(self._buffer)

    @property
    def is_full(self) -> bool:
        """Whether buffer has reached capacity."""
        return self.size >= self.capacity

    def add(self, reading: SensorReading) -> None:
        """Add a reading to the buffer."""
        self._buffer.append(reading)

    def get_latest(self, n: int) -> list[SensorReading]:
        """Get n most recent readings."""
        return list(self._buffer)[-n:]

    def get_all(self) -> list[SensorReading]:
        """Get all readings in buffer."""
        return list(self._buffer)

    def clear(self) -> None:
        """Clear all readings from buffer."""
        self._buffer.clear()
