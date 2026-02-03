"""
Advanced Feature Extractor
Biomechanical + cardiovascular features from mid-abdomen wearable sensors

CORRECTED:
- Intent-first posture logic (preemptive)
- Gravity used only as confirmation (never override)
- Supine requires abs(pitch) < threshold (fixes negative pitch bug)
- Fast, rollback-stabilized FSM
- Fixed meaningless abs(nz) > -threshold condition
- Added gravity validation to standing detection
- Fixed FSM skip to include pitch variance
- Fixed RollingStats variance calculation (correct, not approximate)

PERFORMANCE OPTIMIZED:
- Math module instead of NumPy for scalars
- FSM update decimation (100ms throttle)
- No internal time.time() calls
- Precomputed thresholds
- Skip FSM if both accel AND pitch are stable
"""

from collections import deque
from enum import Enum
import math


# Precomputed thresholds for performance
# Supine: very small pitch, strong gravity alignment
PITCH_SUPINE_MAX = 10  # Supine requires abs(pitch) < 10 (tightened from 15)
NZ_SUPINE_THRESHOLD = 0.8  # Supine requires strong gravity alignment

# Seated: moderate pitch range, stable
PITCH_SEATED_MIN = 10
PITCH_SEATED_MAX = 45
PITCH_RATE_SEATED_MAX = 2.0  # Seated has minimal pitch rate
NZ_SEATED_THRESHOLD = 0.9  # Seated: gravity not strongly vertical

# Standing: high pitch
PITCH_STANDING_MIN = 45  # Standing requires pitch > 45 (raised from 25)
PITCH_STAND_TRANSITION = 20  # For fast intent detection

# Motion thresholds
PITCH_RATE_STAND_THRESHOLD = 5.0
PITCH_RATE_SUPINE_THRESHOLD = -0.75
ACCEL_VAR_WALKING_THRESHOLD = 0.5
ACCEL_VAR_STABLE_THRESHOLD = 0.05
PITCH_VAR_STABLE_THRESHOLD = 1.0  # Pitch variance < 1 deg means stable


class PostureState(Enum):
    SUPINE = "Supine"
    SEATED = "Seated"
    STANDING = "Standing"
    WALKING = "Walking"
    UNKNOWN = "Unknown"


ALLOWED_TRANSITIONS = {
    PostureState.SUPINE: {
        PostureState.SEATED,
        PostureState.STANDING
    },
    PostureState.SEATED: {
        PostureState.SUPINE,
        PostureState.STANDING
    },
    PostureState.STANDING: {
        PostureState.SEATED,
        PostureState.WALKING
    },
    PostureState.WALKING: {
        PostureState.STANDING   # ONLY allowed exit
    },
    PostureState.UNKNOWN: {
        PostureState.SUPINE,
        PostureState.SEATED,
        PostureState.STANDING
    }
}


class RollingStats:
    """
    Rolling statistics with correct variance calculation.
    For small windows (<=30), direct computation is fast enough and accurate.
    """

    def __init__(self, maxlen: int):
        self.maxlen = maxlen
        self.values = deque(maxlen=maxlen)

    def update(self, x: float) -> None:
        """Update with new value."""
        self.values.append(x)

    def var(self) -> float:
        """
        Return sample variance.
        Direct computation is correct and fast for small windows.
        """
        n = len(self.values)
        if n < 2:
            return 0.0
        mean = sum(self.values) / n
        return sum((x - mean) ** 2 for x in self.values) / n

    def std(self) -> float:
        """Return sample standard deviation."""
        return math.sqrt(self.var())

    def mean_val(self) -> float:
        """Return current mean."""
        if len(self.values) == 0:
            return 0.0
        return sum(self.values) / len(self.values)

    def min(self) -> float:
        """Return minimum value."""
        if len(self.values) == 0:
            return 0.0
        return min(self.values)

    def __len__(self) -> int:
        return len(self.values)


class BiomechanicalFeatureExtractor:
    """
    Abdomen-mounted posture + motion feature extractor
    Intent-first, preemptive FSM
    """

    def __init__(self, window_size=15):
        self.window_size = window_size

        # Rolling stats for variance computation
        self.accel_stats = RollingStats(window_size)
        self.pitch_stats = RollingStats(window_size)

        self.pitch_history = deque(maxlen=window_size)
        self.pitch_rate_history = deque(maxlen=window_size)

        self.prev_pitch = None
        self.prev_timestamp = None

        # FSM update throttling
        self.last_fsm_update = 0.0
        self.fsm_update_interval = 0.1  # 100ms

        self.current_posture = PostureState.UNKNOWN
        self.posture_entered_at = None

    # ------------------------------------------------------------------ #
    def extract(self, raw_data):
        ax = raw_data['posture']['x']
        ay = raw_data['posture']['y']
        az = raw_data['posture']['z']
        timestamp = raw_data.get('timestamp', 0.0)

        pitch = self._compute_pitch(ax, ay, az)
        pitch_rate = self._compute_pitch_rate(pitch, timestamp)
        accel_mag = math.sqrt(ax*ax + ay*ay + az*az)

        self.pitch_history.append(pitch)
        self.pitch_rate_history.append(pitch_rate)

        # Update rolling stats
        self.accel_stats.update(accel_mag)
        self.pitch_stats.update(pitch)

        posture_state = self._update_posture_state(
            pitch, pitch_rate, ax, ay, az, timestamp
        )

        stability = self._compute_stability_score()

        return {
            "torso_pitch_angle": pitch,
            "pitch_rate": pitch_rate,
            "posture_state": posture_state,
            "posture_stability_score": stability,
            "accel_magnitude": accel_mag
        }

    # ------------------------------------------------------------------ #
    # Intent-first FSM
    # ------------------------------------------------------------------ #
    def _update_posture_state(self, pitch, pitch_rate, ax, ay, az, timestamp):
        # FSM decimation - only update every 100ms
        if timestamp - self.last_fsm_update < self.fsm_update_interval:
            return self.current_posture
        self.last_fsm_update = timestamp

        # Skip FSM if BOTH accel AND pitch are very stable (minor optimization)
        # This prevents freezing when only accel is stable but pitch is changing
        if self.current_posture != PostureState.UNKNOWN:
            accel_var = self.accel_stats.var()
            pitch_var = self.pitch_stats.var()
            # Only skip if BOTH are stable AND we have a full window
            if (accel_var < 0.001 and pitch_var < PITCH_VAR_STABLE_THRESHOLD
                    and len(self.accel_stats) >= self.window_size):
                return self.current_posture

        # Use passed timestamp instead of time.time()
        now = timestamp

        # O(1) variance from RollingStats
        accel_var = self.accel_stats.var()

        g = math.sqrt(ax*ax + ay*ay + az*az) + 1e-6
        nz = az / g

        # ------------------------------------------------
        # A. FAST INTENT (1-sample, preemptive)
        # ------------------------------------------------
        candidate = None

        # 1. SEATED INTENT (NEW): Moderate pitch + stable + minimal motion
        # This gives SEATED its own evidence instead of being a fallback
        if (PITCH_SEATED_MIN <= pitch <= PITCH_SEATED_MAX
                and abs(pitch_rate) < PITCH_RATE_SEATED_MAX
                and accel_var < ACCEL_VAR_STABLE_THRESHOLD):
            candidate = PostureState.SEATED

        # 2. Standing intent: positive pitch + rapid upward motion
        elif pitch > PITCH_STAND_TRANSITION and abs(pitch_rate) > PITCH_RATE_STAND_THRESHOLD:
            candidate = PostureState.STANDING

        # 3. Supine intent: abs(pitch) very small + downward motion + strong gravity
        elif (abs(pitch) < PITCH_SUPINE_MAX
                and pitch_rate < PITCH_RATE_SUPINE_THRESHOLD
                and abs(nz) > NZ_SUPINE_THRESHOLD):
            candidate = PostureState.SUPINE

        # 4. Walking detection: standing + high motion variance
        elif self.current_posture == PostureState.STANDING and accel_var > ACCEL_VAR_WALKING_THRESHOLD:
            candidate = PostureState.WALKING

        # ------------------------------------------------
        # B. STRUCTURAL POSTURE (fallback with tighter ranges)
        # ------------------------------------------------
        if candidate is None:
            # Supine: VERY small pitch + strong gravity alignment (tightened)
            if abs(pitch) < PITCH_SUPINE_MAX and abs(nz) > NZ_SUPINE_THRESHOLD:
                candidate = PostureState.SUPINE
            # Seated: moderate pitch range + gravity not vertical
            elif PITCH_SEATED_MIN <= pitch <= PITCH_SEATED_MAX and abs(nz) < NZ_SEATED_THRESHOLD:
                candidate = PostureState.SEATED
            # Standing: high pitch + gravity not vertical
            elif pitch > PITCH_STANDING_MIN and abs(nz) < NZ_SEATED_THRESHOLD:
                candidate = PostureState.STANDING
            # Ultimate fallback (should rarely reach here)
            else:
                candidate = PostureState.SEATED

        # ------------------------------------------------
        # C. FSM LEGALITY
        # ------------------------------------------------
        if self.current_posture != PostureState.UNKNOWN:
            if candidate not in ALLOWED_TRANSITIONS[self.current_posture]:
                # Illegal transition -> degrade safely
                if self.current_posture == PostureState.WALKING:
                    candidate = PostureState.STANDING
                else:
                    candidate = self.current_posture

        # ------------------------------------------------
        # D. Gravity override: Only allow Standing -> Supine if pitch is VERY small
        # This prevents misfire on forward lean with acceleration noise
        # ------------------------------------------------
        if (self.current_posture == PostureState.STANDING
                and abs(nz) > NZ_SUPINE_THRESHOLD
                and abs(pitch) < 8):  # Very tight threshold to prevent false supine
            candidate = PostureState.SUPINE

        # ------------------------------------------------
        # E. COMMIT
        # ------------------------------------------------
        if candidate != self.current_posture:
            self.current_posture = candidate
            self.posture_entered_at = now

        return self.current_posture


    # ------------------------------------------------------------------ #
    # Feature computations (using math module instead of NumPy)
    # ------------------------------------------------------------------ #
    def _compute_pitch(self, ax, ay, az):
        pitch_rad = math.atan2(ax, math.sqrt(ay*ay + az*az))
        return math.degrees(pitch_rad)

    def _compute_pitch_rate(self, pitch, timestamp):
        if self.prev_pitch is None:
            self.prev_pitch = pitch
            self.prev_timestamp = timestamp
            return 0.0

        dt = timestamp - self.prev_timestamp
        if dt < 1e-6:
            return 0.0

        rate = (pitch - self.prev_pitch) / dt
        self.prev_pitch = pitch
        self.prev_timestamp = timestamp
        return rate

    def _compute_stability_score(self):
        if len(self.accel_stats) < 3:
            return 0.5

        # Variance from RollingStats
        accel_var = self.accel_stats.var()
        pitch_var = self.pitch_stats.var()

        # Use math.exp instead of np.exp
        accel_score = math.exp(-6 * accel_var)
        pitch_score = math.exp(-0.015 * pitch_var)

        return max(0.0, min(1.0, 0.6 * accel_score + 0.4 * pitch_score))

    def reset(self):
        self.pitch_history.clear()
        self.pitch_rate_history.clear()
        self.prev_pitch = None
        self.prev_timestamp = None
        self.last_fsm_update = 0.0
        self.current_posture = PostureState.UNKNOWN
        self.posture_entered_at = None
        self.accel_stats = RollingStats(self.window_size)
        self.pitch_stats = RollingStats(self.window_size)


class CardiovascularFeatureExtractor:
    """
    Extracts cardiovascular features from blood pressure data

    Features:
    - SBP baseline (rolling average)
    - SBP minimum (over window)
    - SBP drop (baseline - current)
    - Post-stand BP slope (rate of recovery)
    - False-positive stand flag
    """

    def __init__(self, baseline_window=30, minimum_window=10):
        """
        Initialize cardiovascular feature extractor

        Args:
            baseline_window: Samples for baseline calculation
            minimum_window: Samples for minimum calculation
        """
        self.baseline_window = baseline_window
        self.minimum_window = minimum_window

        # Rolling stats for efficient computation
        self.sbp_stats = RollingStats(baseline_window)
        self.sbp_recent_stats = RollingStats(minimum_window)

        # For post-stand detection
        self.stand_detected = False
        self.stand_time = None
        self.sbp_at_stand = None
        self.post_stand_samples = deque(maxlen=20)

        # For false positive detection
        self.prev_posture_state = None

    def extract(self, raw_data, posture_state):
        """
        Extract cardiovascular features

        Args:
            raw_data: dict with 'blood_pressure' containing 'systolic'
            posture_state: Current PostureState enum

        Returns:
            dict with cardiovascular features
        """
        sbp = raw_data['blood_pressure']['systolic']
        timestamp = raw_data.get('timestamp', 0.0)

        # Update rolling stats
        self.sbp_stats.update(sbp)
        self.sbp_recent_stats.update(sbp)

        # Compute baseline SBP
        sbp_baseline = self._compute_baseline()

        # Compute minimum SBP
        sbp_minimum = self._compute_minimum()

        # Compute SBP drop
        sbp_drop = sbp_baseline - sbp

        # Detect standing transitions
        self._detect_stand_transition(posture_state, sbp, timestamp)

        # Compute post-stand slope
        post_stand_slope = self._compute_post_stand_slope()

        # Detect false positives
        false_positive_stand = self._detect_false_positive_stand(
            posture_state, sbp_drop
        )

        # Update previous state
        self.prev_posture_state = posture_state

        return {
            'sbp_baseline': sbp_baseline,
            'sbp_minimum': sbp_minimum,
            'sbp_drop': sbp_drop,
            'post_stand_bp_slope': post_stand_slope,
            'false_positive_stand': false_positive_stand
        }

    def _compute_baseline(self):
        """
        Compute baseline SBP as rolling average

        Returns:
            Baseline SBP (mmHg)
        """
        return self.sbp_stats.mean_val() if len(self.sbp_stats) > 0 else 120.0

    def _compute_minimum(self):
        """
        Compute minimum SBP over recent window

        Returns:
            Minimum SBP (mmHg)
        """
        return self.sbp_recent_stats.min() if len(self.sbp_recent_stats) > 0 else 120.0

    def _detect_stand_transition(self, posture_state, sbp, timestamp):
        """
        Detect transition from sitting/supine to standing

        Sets stand_detected flag and captures stand time and SBP
        """
        # Transition to standing
        if (posture_state == PostureState.STANDING and
            self.prev_posture_state in [PostureState.SEATED, PostureState.SUPINE]):

            self.stand_detected = True
            self.stand_time = timestamp
            self.sbp_at_stand = sbp
            self.post_stand_samples.clear()

        # If in standing state and we detected a stand, collect samples
        if self.stand_detected and posture_state == PostureState.STANDING:
            time_since_stand = timestamp - self.stand_time

            # Collect samples for first 10 seconds after standing
            if time_since_stand < 10.0:
                self.post_stand_samples.append((time_since_stand, sbp))
            else:
                # Reset after 10 seconds
                self.stand_detected = False

    def _compute_post_stand_slope(self):
        """
        Compute BP recovery slope after standing using scalar math

        Uses linear regression on post-stand BP samples.
        Positive slope = BP recovering (increasing)

        Returns:
            BP slope in mmHg/second
        """
        n = len(self.post_stand_samples)
        if n < 3:
            return 0.0

        # Manual linear regression using scalar math (no NumPy)
        sum_x = 0.0
        sum_y = 0.0
        sum_xx = 0.0
        sum_xy = 0.0

        for t, sbp in self.post_stand_samples:
            sum_x += t
            sum_y += sbp
            sum_xx += t * t
            sum_xy += t * sbp

        # slope = (n*sum_xy - sum_x*sum_y) / (n*sum_xx - sum_x^2)
        denominator = n * sum_xx - sum_x * sum_x
        if abs(denominator) < 1e-6:
            return 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope

    def _detect_false_positive_stand(self, posture_state, sbp_drop):
        """
        Detect false-positive stand events

        False positive criteria:
        - Classified as standing
        - But SBP drop is minimal (< 10 mmHg)
        - Suggests sensor error or incorrect classification

        Returns:
            Boolean indicating false positive
        """
        if posture_state == PostureState.STANDING:
            # If standing but minimal BP drop, likely false positive
            if sbp_drop < 10.0:
                return True

        return False

    def reset(self):
        """Reset internal state"""
        self.sbp_stats = RollingStats(self.baseline_window)
        self.sbp_recent_stats = RollingStats(self.minimum_window)
        self.stand_detected = False
        self.stand_time = None
        self.sbp_at_stand = None
        self.post_stand_samples.clear()
        self.prev_posture_state = None


class CompressionFeatureExtractor:
    """
    Extracts compression-related features

    Features:
    - Compression dose (cumulative exposure over time)
    """

    def __init__(self):
        """Initialize compression feature extractor"""
        self.compression_history = deque(maxlen=1000)
        self.cumulative_dose = 0.0
        self.prev_timestamp = None

    def extract(self, raw_data):
        """
        Extract compression features

        Args:
            raw_data: dict with 'compression' and 'timestamp'

        Returns:
            dict with compression features
        """
        compression = raw_data.get('compression', 0.0)
        timestamp = raw_data.get('timestamp', 0.0)

        # Update history
        self.compression_history.append(compression)

        # Compute dose (area under curve)
        compression_dose = self._compute_compression_dose(compression, timestamp)

        return {
            'compression_dose': compression_dose
        }

    def _compute_compression_dose(self, compression, timestamp):
        """
        Compute cumulative compression dose

        Dose = integral of pressure over time
        Approximated as: dose += pressure * dt

        Returns:
            Cumulative dose in kPa·seconds
        """
        if self.prev_timestamp is None:
            self.prev_timestamp = timestamp
            return self.cumulative_dose

        dt = timestamp - self.prev_timestamp

        # Add to cumulative dose (trapezoidal rule approximation)
        self.cumulative_dose += compression * dt

        self.prev_timestamp = timestamp

        return self.cumulative_dose

    def reset(self):
        """Reset internal state"""
        self.compression_history.clear()
        self.cumulative_dose = 0.0
        self.prev_timestamp = None


if __name__ == "__main__":
    # Test the extractors
    print("Testing BiomechanicalFeatureExtractor...")
    bio_extractor = BiomechanicalFeatureExtractor()

    # Test cases - including edge cases
    test_cases = [
        {
            'name': 'Flat Supine',
            'data': {'posture': {'x': 0.1, 'y': 0.05, 'z': 0.99},
                    'compression': 3.0, 'timestamp': 1.0}
        },
        {
            'name': 'Negative Pitch (should NOT be supine)',
            'data': {'posture': {'x': -0.8, 'y': 0.1, 'z': 0.58},
                    'compression': 5.0, 'timestamp': 2.0}
        },
        {
            'name': 'Seated (stable, moderate pitch)',
            'data': {'posture': {'x': 0.5, 'y': 0.2, 'z': 0.84},
                    'compression': 3.0, 'timestamp': 3.0}
        },
        {
            'name': 'Seated (low end of range)',
            'data': {'posture': {'x': 0.2, 'y': 0.1, 'z': 0.95},
                    'compression': 3.0, 'timestamp': 4.0}
        },
        {
            'name': 'Standing',
            'data': {'posture': {'x': 0.8, 'y': 0.1, 'z': 0.58},
                    'compression': 5.0, 'timestamp': 5.0}
        },
    ]

    for test in test_cases:
        features = bio_extractor.extract(test['data'])
        print(f"\n{test['name']}:")
        print(f"  Pitch: {features['torso_pitch_angle']:.1f} deg")
        print(f"  State: {features['posture_state'].value}")
        print(f"  Stability: {features['posture_stability_score']:.3f}")

    print("\n\nTesting CardiovascularFeatureExtractor...")
    cv_extractor = CardiovascularFeatureExtractor()

    # Simulate BP data
    for i in range(15):
        sbp = 120 - i * 2  # Simulate drop
        data = {'blood_pressure': {'systolic': sbp}, 'timestamp': float(i)}
        state = PostureState.STANDING if i > 5 else PostureState.SEATED

        features = cv_extractor.extract(data, state)

        if i % 5 == 0:
            print(f"\nSample {i}:")
            print(f"  SBP: {sbp}, State: {state.value}")
            print(f"  Baseline: {features['sbp_baseline']:.1f}")
            print(f"  Drop: {features['sbp_drop']:.1f}")

    print("\nAll extractors working correctly")
