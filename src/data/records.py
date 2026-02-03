"""
Data models for the physiological monitoring system.

Defines records for per-stand instances and daily aggregates,
including BP recovery metrics and pattern recognition features.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Optional, Union
from pathlib import Path
import json


class PostureState(Enum):
    """Detected posture states."""
    SUPINE = "supine"
    SEATED = "seated"
    STANDING = "standing"
    WALKING = "walking"
    TRANSITIONING = "transitioning"
    UNKNOWN = "unknown"


class FeedbackType(Enum):
    """User feedback categories."""
    NONE = "none"
    DISCOMFORT = "discomfort"           # Too strong
    WEAK = "weak"                       # Not strong enough
    ACCEPTABLE = "acceptable"           # Felt appropriate


@dataclass
class BloodPressureReading:
    """
    Blood pressure reading with recovery metrics.

    Tracks BP before, during, and after a stand event.
    """
    baseline_sys: float      # Baseline systolic (mmHg) - before stand
    baseline_dia: float      # Baseline diastolic (mmHg)
    minimum_sys: float       # Minimum systolic during/after stand (mmHg)
    minimum_dia: float       # Minimum diastolic during/after stand (mmHg)
    drop_sys: float          # Systolic drop = baseline - minimum (mmHg)
    drop_dia: float          # Diastolic drop = baseline - minimum (mmHg)
    slope_sys: float         # Rate of systolic change (mmHg/s)
    slope_dia: float         # Rate of diastolic change (mmHg/s)
    recovery_time: float     # Time to return to baseline (seconds)
    time_to_minimum: float   # Time from stand to minimum BP (seconds)
    timestamp_minimum: float # Timestamp of minimum BP


@dataclass
class CompressionDelivery:
    """
    Compression parameters delivered during a stand event.
    """
    pressure: float          # Applied pressure (mmHg)
    duration: float          # Compression duration (seconds)
    dose: float              # Total dose = pressure * duration (mmHg·s)
    target_pressure: float   # Target pressure requested (mmHg)
    adaptive_factor: float   # Multiplier applied (0.5-2.0)
    reason: str              # Why this strength was chosen


@dataclass
class StandEventRecord:
    """
    Complete record for a single sit-to-stand event instance.

    All data collected for one stand event, including BP recovery,
    compression delivery, posture, and classification results.
    """
    # === Event Metadata ===
    event_id: str                           # Unique identifier
    timestamp: float                        # Unix timestamp of stand start
    date: str                               # ISO date string (YYYY-MM-DD)
    time_of_day: str                        # Time string (HH:MM:SS)

    # === Posture Detection ===
    posture_before: PostureState            # Posture before stand
    posture_after: PostureState             # Posture after stand
    posture_confidence: float               # Classification confidence (0-1)

    # === Stand Detection ===
    stand_duration: float                   # Duration of transition (s)
    peak_acceleration: float                # Peak upward accel (m/s²)
    is_false_positive: bool                 # Flagged as false positive

    # === Blood Pressure ===
    bp_reading: Optional[BloodPressureReading] = None

    # === Compression Delivery ===
    compression: Optional[CompressionDelivery] = None

    # === Pattern Recognition ===
    is_first_stand_of_day: bool = False     # First stand detected today
    is_frequent_cycle: bool = False         # Part of frequent sit-stand pattern
    stands_in_last_hour: int = 0            # Count of stands in past hour
    minutes_since_previous_stand: Optional[float] = None

    # === Classification ===
    predicted_response: str = "unknown"     # Expected BP response category
    classifier_confidence: float = 0.0      # ML model confidence (0-1)

    # === User Feedback ===
    user_feedback: FeedbackType = FeedbackType.NONE
    feedback_notes: str = ""

    # === Raw Data References ===
    imu_data_path: Optional[str] = None     # Path to raw IMU data file
    bp_data_path: Optional[str] = None      # Path to raw BP data file

    def to_json(self, path: Optional[Union[Path, str]] = None) -> str:
        """
        Serialize to JSON.

        Args:
            path: If provided, write to file. Otherwise return string.
        """
        # Convert to dict, handling enums and nested objects
        data = {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "date": self.date,
            "time_of_day": self.time_of_day,
            "posture_before": self.posture_before.value,
            "posture_after": self.posture_after.value,
            "posture_confidence": self.posture_confidence,
            "stand_duration": self.stand_duration,
            "peak_acceleration": self.peak_acceleration,
            "is_false_positive": self.is_false_positive,
            "is_first_stand_of_day": self.is_first_stand_of_day,
            "is_frequent_cycle": self.is_frequent_cycle,
            "stands_in_last_hour": self.stands_in_last_hour,
            "minutes_since_previous_stand": self.minutes_since_previous_stand,
            "predicted_response": self.predicted_response,
            "classifier_confidence": self.classifier_confidence,
            "user_feedback": self.user_feedback.value,
            "feedback_notes": self.feedback_notes,
            "imu_data_path": self.imu_data_path,
            "bp_data_path": self.bp_data_path,
        }

        # Add nested objects
        if self.bp_reading:
            data["bp_reading"] = asdict(self.bp_reading)
        if self.compression:
            data["compression"] = asdict(self.compression)

        json_str = json.dumps(data, indent=2)

        if path:
            Path(path).write_text(json_str)

        return json_str

    @classmethod
    def from_json(cls, json_str: str) -> "StandEventRecord":
        """Deserialize from JSON string."""
        data = json.loads(json_str)

        # Reconstruct enums
        data["posture_before"] = PostureState(data["posture_before"])
        data["posture_after"] = PostureState(data["posture_after"])
        data["user_feedback"] = FeedbackType(data["user_feedback"])

        # Reconstruct nested objects
        if "bp_reading" in data and data["bp_reading"]:
            data["bp_reading"] = BloodPressureReading(**data["bp_reading"])
        else:
            data["bp_reading"] = None

        if "compression" in data and data["compression"]:
            data["compression"] = CompressionDelivery(**data["compression"])
        else:
            data["compression"] = None

        return cls(**data)


@dataclass
class DailyAggregate:
    """
    Daily summary statistics.

    Aggregates all stand events for a single day.
    """
    # === Date ===
    date: str                                 # ISO date (YYYY-MM-DD)
    timestamp: float                          # Unix timestamp of start of day

    # === Stand Counts ===
    total_stands_detected: int = 0
    valid_stands: int = 0                     # Excluding false positives
    false_positives: int = 0

    # === Temporal Patterns ===
    first_stand_time: Optional[str] = None    # Time of first stand (HH:MM:SS)
    last_stand_time: Optional[str] = None     # Time of last stand (HH:MM:SS)
    first_stand_of_day_detected: bool = False

    # === Frequent Cycling ===
    frequent_cycle_count: int = 0             # Stands part of frequent cycles
    avg_interval_between_stands: float = 0.0  # Average minutes between stands
    max_stands_in_hour: int = 0               # Peak frequency

    # === Blood Pressure Aggregates ===
    avg_bp_drop_sys: float = 0.0              # Average systolic drop (mmHg)
    avg_bp_drop_dia: float = 0.0              # Average diastolic drop (mmHg)
    max_bp_drop_sys: float = 0.0              # Maximum systolic drop
    max_recovery_time: float = 0.0            # Longest recovery time (s)
    avg_recovery_time: float = 0.0            # Average recovery time (s)

    # === Compression Summary ===
    total_compression_dose: float = 0.0       # Sum of all doses (mmHg·s)
    avg_compression_pressure: float = 0.0     # Average pressure applied (mmHg)
    total_compressions_delivered: int = 0     # Number of compressions triggered

    # === Feedback Summary ===
    discomfort_count: int = 0                 # Times user reported discomfort
    weak_count: int = 0                       # Times user reported too weak
    acceptable_count: int = 0                 # Times user reported OK

    # === Classification Performance ===
    true_positives: int = 0                   # Correctly identified true stands
    true_negatives: int = 0                   # Correctly identified false positives
    false_positives_made: int = 0             # Incorrectly flagged as false positive
    false_negatives_made: int = 0             # Missed false positives

    def to_json(self, path: Optional[Union[Path, str]] = None) -> str:
        """Serialize to JSON."""
        data = asdict(self)
        json_str = json.dumps(data, indent=2)

        if path:
            Path(path).write_text(json_str)

        return json_str

    @classmethod
    def from_records(cls, records: list[StandEventRecord]) -> "DailyAggregate":
        """
        Create daily aggregate from list of stand records.

        Args:
            records: List of StandEventRecord for the day

        Returns:
            DailyAggregate with computed statistics
        """
        if not records:
            raise ValueError("Cannot create aggregate from empty records")

        date = records[0].date
        timestamp = datetime.fromisoformat(date).timestamp()

        # Count valid stands
        valid = [r for r in records if not r.is_false_positive]

        # Temporal
        times = [r.time_of_day for r in valid]
        first_time = min(times) if times else None
        last_time = max(times) if times else None

        # Intervals between stands
        intervals = []
        for i in range(1, len(valid)):
            if valid[i-1].minutes_since_previous_stand is not None:
                intervals.append(valid[i-1].minutes_since_previous_stand)

        avg_interval = sum(intervals) / len(intervals) if intervals else 0

        # Frequent cycling
        frequent_count = sum(1 for r in valid if r.is_frequent_cycle)

        # Max stands in hour
        hourly_counts = {}
        for r in valid:
            hour = r.time_of_day[:2]  # HH
            hourly_counts[hour] = hourly_counts.get(hour, 0) + 1
        max_per_hour = max(hourly_counts.values()) if hourly_counts else 0

        # BP aggregates
        bp_drops_sys = [r.bp_reading.drop_sys for r in valid if r.bp_reading]
        bp_drops_dia = [r.bp_reading.drop_dia for r in valid if r.bp_reading]
        recovery_times = [r.bp_reading.recovery_time for r in valid if r.bp_reading and r.bp_reading.recovery_time > 0]

        avg_drop_sys = sum(bp_drops_sys) / len(bp_drops_sys) if bp_drops_sys else 0
        avg_drop_dia = sum(bp_drops_dia) / len(bp_drops_dia) if bp_drops_dia else 0
        max_drop_sys = max(bp_drops_sys) if bp_drops_sys else 0
        max_recov = max(recovery_times) if recovery_times else 0
        avg_recov = sum(recovery_times) / len(recovery_times) if recovery_times else 0

        # Compression
        compressions = [r.compression for r in valid if r.compression]
        total_dose = sum(c.dose for c in compressions) if compressions else 0
        avg_pressure = sum(c.pressure for c in compressions) / len(compressions) if compressions else 0

        # Feedback
        discomfort = sum(1 for r in valid if r.user_feedback == FeedbackType.DISCOMFORT)
        weak = sum(1 for r in valid if r.user_feedback == FeedbackType.WEAK)
        acceptable = sum(1 for r in valid if r.user_feedback == FeedbackType.ACCEPTABLE)

        return cls(
            date=date,
            timestamp=timestamp,
            total_stands_detected=len(records),
            valid_stands=len(valid),
            false_positives=sum(1 for r in records if r.is_false_positive),
            first_stand_time=first_time,
            last_stand_time=last_time,
            first_stand_of_day_detected=any(r.is_first_stand_of_day for r in valid),
            frequent_cycle_count=frequent_count,
            avg_interval_between_stands=avg_interval,
            max_stands_in_hour=max_per_hour,
            avg_bp_drop_sys=avg_drop_sys,
            avg_bp_drop_dia=avg_drop_dia,
            max_bp_drop_sys=max_drop_sys,
            max_recovery_time=max_recov,
            avg_recovery_time=avg_recov,
            total_compression_dose=total_dose,
            avg_compression_pressure=avg_pressure,
            total_compressions_delivered=len(compressions),
            discomfort_count=discomfort,
            weak_count=weak,
            acceptable_count=acceptable,
        )


def create_stand_record_id(timestamp: float) -> str:
    """Create unique ID for a stand record from timestamp."""
    dt = datetime.fromtimestamp(timestamp)
    return f"stand_{dt.strftime('%Y%m%d_%H%M%S')}"
