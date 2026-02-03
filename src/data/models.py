"""
Data models and types for the physiological monitoring system.

Defines all data structures used throughout the system for clarity and type safety.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import numpy as np


class PostureState(Enum):
    """Detected posture states."""
    UNKNOWN = "unknown"
    SITTING = "sitting"
    STANDING = "standing"
    LYING = "lying"
    WALKING = "walking"
    TRANSITIONING = "transitioning"


class SystemState(Enum):
    """
    High-level system state for detection behavior.

    Controls when stand detection is active and how events are interpreted.
    """
    DAY_INIT = "day_init"  # Initial period after system start (first 30 min)
    ACTIVE_DAY = "active_day"  # Normal daytime operation
    NIGHT = "night"  # Nighttime (supine for extended period)


class CompressionState(Enum):
    """Compression device states."""
    OFF = "off"
    INFLATING = "inflating"
    COMPRESSING = "compressing"
    DEFLATING = "deflating"


@dataclass
class IMUSample:
    """
    Single IMU sensor reading.

    Accelerometer and gyroscope data from a single time point.
    All values in SI units (m/s² for accel, rad/s for gyro).
    """
    timestamp: float  # Unix timestamp (seconds)
    accel_x: float
    accel_y: float
    accel_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float

    @property
    def accel_vector(self) -> np.ndarray:
        """Acceleration as numpy array."""
        return np.array([self.accel_x, self.accel_y, self.accel_z])

    @property
    def gyro_vector(self) -> np.ndarray:
        """Gyroscope data as numpy array."""
        return np.array([self.gyro_x, self.gyro_y, self.gyro_z])

    @property
    def accel_magnitude(self) -> float:
        """Magnitude of acceleration vector."""
        return float(np.linalg.norm(self.accel_vector))


@dataclass
class BloodPressureSample:
    """
    Blood pressure sensor reading.

    Systolic/diastolic pressures in mmHg, heart rate in BPM.
    """
    timestamp: float  # Unix timestamp (seconds)
    systolic: float  # mmHg
    diastolic: float  # mmHg
    heart_rate: float  # BPM
    mean_arterial_pressure: Optional[float] = None

    def __post_init__(self):
        """Calculate mean arterial pressure if not provided."""
        if self.mean_arterial_pressure is None:
            # MAP ≈ DP + 1/3(SP - DP)
            self.mean_arterial_pressure = self.diastolic + (self.systolic - self.diastolic) / 3


@dataclass
class CompressionSample:
    """
    Compression device reading.

    Pressure in mmHg, state indicates current device phase.
    """
    timestamp: float  # Unix timestamp (seconds)
    pressure: float  # mmHg
    state: CompressionState
    chamber: Optional[int] = None  # Chamber ID (if multi-chamber device)


@dataclass
class SensorReading:
    """
    Combined sensor reading at a single time point.

    Aggregates all sensor data for synchronized processing.
    """
    timestamp: float
    imu: Optional[IMUSample] = None
    blood_pressure: Optional[BloodPressureSample] = None
    compression: Optional[CompressionSample] = None

    @classmethod
    def from_timestamp(cls, timestamp: float) -> "SensorReading":
        """Create empty reading with just timestamp."""
        return cls(timestamp=timestamp)


@dataclass
class StandEvent:
    """
    Detected stand event with classification.

    Represents a single sit-to-stand transition with ML classification.
    """
    start_time: float  # When stand began (transition start)
    peak_time: float  # Time of maximum vertical acceleration
    end_time: float  # When settled in standing position
    is_valid: bool  # ML classification (True = real stand, False = false positive)
    confidence: float  # Classification confidence (0-1)
    peak_acceleration: float  # Peak vertical accel during event (m/s²)
    features: dict = field(default_factory=dict)  # Extracted features for explainability
    is_first_stand_of_day: bool = False  # First confirmed stand after DAY_INIT or night
    is_false_positive_flagged: bool = False  # Flagged as false positive by timeout
    compression_triggered: bool = False  # Whether compression was triggered

    @property
    def duration(self) -> float:
        """Duration of stand event in seconds."""
        return self.end_time - self.start_time

    @property
    def label(self) -> str:
        """Human-readable label."""
        if self.is_false_positive_flagged:
            return "FALSE_POSITIVE (TIMEOUT)"
        return "VALID_STAND" if self.is_valid else "FALSE_POSITIVE"


@dataclass
class DailyAggregate:
    """
    Daily summary statistics.

    Aggregated metrics for a single day.
    """
    date: str  # ISO date string (YYYY-MM-DD)
    total_stands: int
    valid_stands: int
    false_positives: int
    avg_stand_duration: float  # seconds
    avg_peak_acceleration: float  # m/s²
    total_compression_time: float  # seconds
    avg_bp_systolic: Optional[float] = None  # mmHg
    avg_bp_diastolic: Optional[float] = None  # mmHg
    avg_heart_rate: Optional[float] = None  # BPM


@dataclass
class StandLogEntry:
    """
    Detailed log entry for a single stand event.

    Written to per-stand log file for traceability.
    """
    event_id: str  # Unique identifier
    timestamp: datetime  # When event occurred
    is_valid: bool
    confidence: float
    duration: float
    peak_acceleration: float
    features: dict
    compression_triggered: bool
    notes: str = ""
