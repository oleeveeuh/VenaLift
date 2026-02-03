"""
Unified State Machine Module (Physiologically Robust + Noise Filtering)

Implements time-based, hysteresis-aware state machines with noise rejection for:
1. Device operational mode
2. Stand event confirmation & monitoring
3. Sensor calibration

All state transitions are debounced and require sustained conditions.
Noise filtering added via sliding windows and signal smoothing.
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict
from collections import deque
import time


# ---------------------------------------------------------------------
# ENUMS
# ---------------------------------------------------------------------

class DeviceMode(Enum):
    OFF = "OFF"
    DAY_INIT = "DAY_INIT"
    ACTIVE_DAY = "ACTIVE_DAY"
    PASSIVE = "PASSIVE"


class StandState(Enum):
    IDLE = "IDLE"
    CONFIRMING = "CONFIRMING"
    MONITORING = "MONITORING"
    REFRACTORY = "REFRACTORY"


class CalibrationState(Enum):
    NOT_ACTIVE = "NOT_ACTIVE"
    BASELINE_COLLECTION = "BASELINE_COLLECTION"
    VALIDATION = "VALIDATION"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


# ---------------------------------------------------------------------
# DEVICE MODE STATE MACHINE (NOISE ROBUST)
# ---------------------------------------------------------------------

class DeviceModeStateMachine:
    """
    Controls high-level device lifecycle using sustained activity.
    
    NOW WITH NOISE FILTERING:
    - Sliding window majority voting
    - Grace periods for brief interruptions
    - Separate inactivity tracking
    
    OFF → DAY_INIT → ACTIVE_DAY ⇄ PASSIVE → OFF
    """

    DAY_INIT_MIN_DURATION = 30.0        # seconds
    INACTIVITY_TIMEOUT = 60.0           # seconds
    ACTIVITY_CONFIRM_TIME = 5.0         # seconds of sustained activity (increased)
    ACTIVITY_GRACE_PERIOD = 3.0         # allow brief interruptions

    def __init__(self):
        self.state = DeviceMode.OFF
        self.state_entered_at = time.time()

        self._last_motion_time = time.time()
        self._activity_start_time: Optional[float] = None
        self._inactivity_start_time: Optional[float] = None
        
        # NOISE FILTERING: sliding window majority voting
        self._activity_window = deque(maxlen=10)  # last 10 readings
        self._activity_threshold = 6  # need 6/10 to count as "active"

        self.history = deque(maxlen=100)

    def update(self, *, has_activity: bool) -> None:
        now = time.time()

        # Add to sliding window for noise filtering
        self._activity_window.append(has_activity)
        
        # Filtered activity: require majority of recent samples to be active
        filtered_active = sum(self._activity_window) >= self._activity_threshold


        
        # Track sustained activity with grace period for brief interruptions
        if filtered_active:
            if self._activity_start_time is None:
                self._activity_start_time = now
            self._last_motion_time = now
            self._inactivity_start_time = None  # reset inactivity counter
        else:
            # Allow brief interruptions before resetting activity
            if self._activity_start_time is not None:
                if now - self._last_motion_time > self.ACTIVITY_GRACE_PERIOD:
                    self._activity_start_time = None
            
            # Start tracking inactivity
            if self._inactivity_start_time is None:
                self._inactivity_start_time = now

        # DAY_INIT → ACTIVE_DAY
        if self.state == DeviceMode.DAY_INIT:
            if now - self.state_entered_at >= self.DAY_INIT_MIN_DURATION:
                self._transition(DeviceMode.ACTIVE_DAY, "Initialization complete")

        # ACTIVE_DAY → PASSIVE (sustained inactivity)
        elif self.state == DeviceMode.ACTIVE_DAY:
            if (self._inactivity_start_time is not None and 
                now - self._inactivity_start_time >= self.INACTIVITY_TIMEOUT):
                self._transition(DeviceMode.PASSIVE, "Sustained inactivity")

        # PASSIVE → ACTIVE_DAY (sustained activity)
        elif self.state == DeviceMode.PASSIVE:
            if (
                self._activity_start_time is not None
                and now - self._activity_start_time >= self.ACTIVITY_CONFIRM_TIME
            ):
                self._transition(DeviceMode.ACTIVE_DAY, "Sustained activity detected")

    def set_state(self, new_state: DeviceMode, reason: str = "") -> None:
        self._transition(new_state, reason)

    def _transition(self, new_state: DeviceMode, reason: str) -> None:
        if new_state == self.state:
            return

        self.history.append({
            "time": time.time(),
            "from": self.state.value,
            "to": new_state.value,
            "reason": reason
        })
        self.state = new_state
        self.state_entered_at = time.time()

    def get_status(self) -> Dict:
        return {
            "state": self.state.value,
            "duration": time.time() - self.state_entered_at
        }


# ---------------------------------------------------------------------
# STAND CONFIRMATION & MONITORING STATE MACHINE (NOISE ROBUST)
# ---------------------------------------------------------------------

@dataclass
class StandEventData:
    start_time: float
    peak_time: float
    max_sbp_drop: float
    recovery_time: Optional[float]


class StandStateMachine:
    """
    Confirms and monitors a stand event AFTER an IMU trigger.
    
    NOW WITH NOISE FILTERING:
    - Trigger debouncing with sliding window
    - Longer confirmation time
    - SBP drop smoothing
    - Extended refractory period
    """

    CONFIRMATION_TIME = 1.0      # seconds (increased from 0.25)
    MONITOR_DURATION = 10.0      # seconds
    REFRACTORY_PERIOD = 8.0      # seconds (increased from 5)
    RECOVERY_FRACTION = 0.5
    
    # NOISE FILTERING
    TRIGGER_WINDOW_SIZE = 5      # last 5 IMU readings
    TRIGGER_THRESHOLD = 3        # need 3/5 to confirm trigger
    SBP_SMOOTHING_WINDOW = 3     # smooth SBP drops over 3 samples

    def __init__(self):
        self.state = StandState.IDLE
        self.state_entered_at = time.time()

        self.current_event: Optional[StandEventData] = None
        self.stand_count = 0

        self._last_trigger_time: Optional[float] = None
        self._early_event_emitted = False
        
        # NOISE FILTERING
        self._trigger_window = deque(maxlen=self.TRIGGER_WINDOW_SIZE)
        self._sbp_window = deque(maxlen=self.SBP_SMOOTHING_WINDOW)
        
        self.history = deque(maxlen=100)

    def update(
        self,
        *,
        imu_stand_trigger: bool,
        sbp_drop: float,
        timestamp: float
    ) -> Optional[StandEventData | str]:

        now = time.time()
        
        # Add to trigger window
        self._trigger_window.append(imu_stand_trigger)
        # filtered_trigger = sum(self._trigger_window) >= self.TRIGGER_THRESHOLD
        

        filtered_trigger = (
            len(self._trigger_window) == self.TRIGGER_WINDOW_SIZE
            and all(self._trigger_window)
        )

        # Add to SBP smoothing window
        self._sbp_window.append(sbp_drop)
        smoothed_sbp_drop = sum(self._sbp_window) / len(self._sbp_window)

        # -----------------------------
        # IDLE → CONFIRMING
        # -----------------------------
        if self.state == StandState.IDLE:
            if filtered_trigger:  # Use filtered trigger
                self._last_trigger_time = now
                self._enter_confirming(timestamp, smoothed_sbp_drop)

        # -----------------------------
        # CONFIRMING → MONITORING
        # -----------------------------
        elif self.state == StandState.CONFIRMING:
            # EARLY SIGNAL — happens DURING the rise
            if not self._early_event_emitted:
                self._early_event_emitted = True
                return "STAND_INITIATED"

            if now - self.state_entered_at >= self.CONFIRMATION_TIME:
                self._enter_monitoring()

        # -----------------------------
        # MONITORING → REFRACTORY
        # -----------------------------
        elif self.state == StandState.MONITORING:
            self._update_monitoring(smoothed_sbp_drop, timestamp)

            if now - self.state_entered_at >= self.MONITOR_DURATION:
                return self._finalize_event()

        # -----------------------------
        # REFRACTORY → IDLE
        # -----------------------------
        elif self.state == StandState.REFRACTORY:
            if now - self.state_entered_at >= self.REFRACTORY_PERIOD:
                self.state = StandState.IDLE
                self.state_entered_at = now
                # Clear windows when returning to IDLE
                self._trigger_window.clear()
                self._sbp_window.clear()

        return None

    def _enter_confirming(self, timestamp: float, sbp_drop: float):
        self.state = StandState.CONFIRMING
        self.state_entered_at = time.time()
        self.current_event = StandEventData(
            start_time=timestamp,
            peak_time=timestamp,
            max_sbp_drop=sbp_drop,
            recovery_time=None
        )

    def _enter_monitoring(self):
        self.state = StandState.MONITORING
        self.state_entered_at = time.time()
        self.stand_count += 1

    def _update_monitoring(self, sbp_drop: float, timestamp: float):
        if not self.current_event:
            return

        self.current_event.max_sbp_drop = max(
            self.current_event.max_sbp_drop, sbp_drop
        )

        if (
            self.current_event.recovery_time is None
            and sbp_drop <= self.current_event.max_sbp_drop * self.RECOVERY_FRACTION
        ):
            self.current_event.recovery_time = (
                timestamp - self.current_event.start_time
            )

    def _finalize_event(self) -> StandEventData:
        event = self.current_event
        self.current_event = None
        self._early_event_emitted = False   # RESET HERE
        self.state = StandState.REFRACTORY
        self.state_entered_at = time.time()
        return event

    def reset(self):
        self.state = StandState.IDLE
        self.current_event = None
        self.stand_count = 0
        self._trigger_window.clear()
        self._sbp_window.clear()
        self.history.clear()


# ---------------------------------------------------------------------
# CALIBRATION STATE MACHINE (NOISE ROBUST)
# ---------------------------------------------------------------------

class CalibrationStateMachine:
    """
    Manages baseline calibration with stability checks.
    
    NOW WITH NOISE FILTERING:
    - More samples required
    - Outlier rejection
    - Stricter stability thresholds
    """

    BASELINE_SAMPLES = 50           # increased from 30
    MIN_COLLECTION_TIME = 15.0      # increased from 10
    MAX_OUTLIER_FRACTION = 0.1      # reject if >10% outliers

    def __init__(self):
        self.state = CalibrationState.NOT_ACTIVE
        self.state_entered_at = time.time()

        self.sbp_samples = []
        self.pitch_samples = []
        self.message = ""

    def start(self):
        if self.state != CalibrationState.NOT_ACTIVE:
            return False

        self.state = CalibrationState.BASELINE_COLLECTION
        self.state_entered_at = time.time()
        self.sbp_samples.clear()
        self.pitch_samples.clear()
        self.message = ""
        return True

    def update(self, *, sbp: float, pitch: float):
        if self.state == CalibrationState.BASELINE_COLLECTION:
            self.sbp_samples.append(sbp)
            self.pitch_samples.append(pitch)

            if (
                len(self.sbp_samples) >= self.BASELINE_SAMPLES
                and time.time() - self.state_entered_at >= self.MIN_COLLECTION_TIME
            ):
                self._validate()

    def _validate(self):
        import numpy as np

        # Remove outliers using IQR method
        sbp_clean, sbp_outliers = self._remove_outliers(self.sbp_samples)
        pitch_clean, pitch_outliers = self._remove_outliers(self.pitch_samples)
        
        total_outliers = sbp_outliers + pitch_outliers
        outlier_fraction = total_outliers / (2 * len(self.sbp_samples))
        
        if outlier_fraction > self.MAX_OUTLIER_FRACTION:
            self._fail(f"Too many outliers ({outlier_fraction:.1%})")
            return

        sbp_mean = np.mean(sbp_clean)
        sbp_std = np.std(sbp_clean)
        pitch_std = np.std(pitch_clean)

        if not (80 <= sbp_mean <= 180):
            self._fail(f"SBP mean out of range: {sbp_mean:.1f}")
            return

        if sbp_std > 12:  # stricter threshold
            self._fail(f"SBP too variable (σ={sbp_std:.1f})")
            return

        if pitch_std > 8:  # stricter threshold
            self._fail(f"Posture unstable (σ={pitch_std:.1f})")
            return

        self.state = CalibrationState.COMPLETE
        self.message = f"Calibration successful (n={len(sbp_clean)})"

    def _remove_outliers(self, data):
        """Remove outliers using IQR method"""
        import numpy as np
        
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        
        clean = [x for x in data if lower <= x <= upper]
        n_outliers = len(data) - len(clean)
        
        return clean, n_outliers

    def _fail(self, msg: str):
        self.state = CalibrationState.FAILED
        self.message = msg

    def reset(self):
        self.state = CalibrationState.NOT_ACTIVE
        self.sbp_samples.clear()
        self.pitch_samples.clear()
        self.message = ""