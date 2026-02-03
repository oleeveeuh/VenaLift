# Physiological Monitoring System

Built for USC ASBME's Make-a-thon 2026: A real-time physiological monitoring system for detecting sit-to-stand events and tracking mobility patterns. Designed for Parkinson's disease patients and individuals at risk of orthostatic hypotension.

## Dashboards

### Patient Dashboard (`patient_dashboard.py`)

A comforting, patient-friendly interface focused on positive reinforcement and safety reassurance.

**Features:**
- **"How You're Doing"** - Large encouraging status message that updates based on daily progress
- **Today's Activity** - Shows stands completed with progress bar toward daily goal (12 stands)
- **Safety Status** - Reassuring feedback about transition quality (no risky transitions detected)
- **Today's Highlights** - Summary card showing stands today, safe stands count, and stability trend
- **Gentle Reminder** - Context-aware coaching messages that update every 5 minutes:
  - While sitting: "Take a slow breath before standing", "Hydration helps with dizziness"
  - While standing: "You've been steady - excellent balance", "It might be a good time to sit and rest"
- **Recent Stands** - Live list of recent stand events with timestamps
- **Charts** - 7-day activity chart, weekly summary, activity heatmap, and longitudinal mobility trend

**Design Philosophy:**
- Feels like a supportive companion, not a medical device
- Focuses on safety reassurance and positive encouragement
- Avoids clinical terminology and anxiety-inducing metrics

### Clinical Dashboard (`clinical_dashboard.py`)

A comprehensive clinical interface for healthcare providers to monitor patient status and analyze trends.

**Real-Time Monitoring:**
- **Posture State** - Current posture (Supine/Seated/Standing/Walking) with pitch angle and stability score
- **Stand Detection** - Current state (IDLE/CONFIRMING/MONITORING/REFRACTORY)
- **Blood Pressure** - Real-time SBP/DBP/HR with drop detection
- **Compression Therapy** - Device state, pressure levels, cycle count, daily dose

**Clinical Alerts (Risk Stratification):**
| Severity | Alert Type | Trigger | Action |
|----------|-----------|---------|--------|
| 🔴 CRITICAL | Severe BP Drop | SBP drop >30 mmHg | IMMEDIATE attention required |
| 🟠 HIGH | Orthostatic Hypotension | SBP drop >20 mmHg | Review standing protocol |
| 🟡 MEDIUM | Fall Risk | ≥3 severe drops in recent stands | Consider compression adjustment |
| 🟡 MEDIUM | Instability Detected | Stability score <0.3 | Monitor patient closely |
| 🟡 MEDIUM | High Compression Dose | Daily dose >10,000 kPa·s | Review compression schedule |
| 🟡 MEDIUM | Reduced Mobility | Stands <50% of 14-day baseline | Assess patient condition |
| 🟡 MEDIUM | Daily Orthostatic Stress | ≥3 drops ≥20 mmHg today | Consider therapy adjustment |
| 🟡 MEDIUM | Standing Instability | Low stability >30 sec while standing | Evaluate fall risk |

**Quality Metrics:**
- **Stand Quality Metrics** - Average BP drop, detection accuracy, confirmation rate, recovery time
- **Compression Efficacy** - Comparison of BP drops with vs without compression therapy

**Charts & Analytics:**
- Daily stand count with trend overlay
- Weekly activity summary by weekday
- Activity heatmap (hour vs weekday patterns)
- Longitudinal mobility trend (monthly aggregation)
- Standing/supine time distribution

**Export & Controls:**
- Quick export to CSV/JSON
- Mock data generation for testing
- Arduino connection management
- Alert simulation buttons for demo

---

## Detection Logic

### State Machine Overview

The system uses a multi-layered state machine to reliably detect and validate stand events while filtering out noise and false positives.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Stand State Machine                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   IDLE ──────► CONFIRMING ─────► MONITORING ─────► REFRACTORY      │
│       ▲              │                    │              │           │
│       │              │                    │              │           │
│       │              └────────────────────┘              │           │
│       │                       │                       │           │
│       └───────────────────────┴───────────────────────┘           │
│                           (after timeout)                            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### State Transitions Explained

| State | Condition | Duration | Purpose |
|-------|-----------|----------|---------|
| **IDLE** | Waiting for trigger | - | Baseline state, monitoring for posture changes |
| **CONFIRMING** | Valid IMU trigger detected | 1.0 second | Verify the trigger is genuine (debounce) |
| **MONITORING** | Confirmation period complete | 10 seconds | Track BP drop and recovery, collect metrics |
| **REFRACTORY** | Monitoring period complete | 8 seconds | Prevent duplicate detection of same event |

### Trigger Detection (IMU)

Posture detection uses mid-abdomen IMU sensors with an intent-first approach:

**Thresholds:**
- **Supine**: `abs(pitch) < 10°` AND strong gravity alignment (`nz > 0.8`)
- **Seated**: `10° < pitch < 45°` with minimal pitch rate
- **Standing**: `pitch > 45°` with gravity validation
- **Walking**: High pitch with elevated acceleration variance

**Noise Filtering:**
- Sliding window majority voting (5 samples, require all positive)
- SBP drop smoothing over 3 samples
- Extended refractory period prevents re-detection

**Key Principle:** Gravity alignment confirms but never overrides intent. Posture transitions are detected by pitch rate exceeding threshold (5 deg/s for standing, -0.75 deg/s for supine).

### Event Confirmation Flow

```
1. IMU detects standing posture (pitch > 45°)
                    ↓
2. Trigger enters confirmation window (5-sample sliding window)
                    ↓
3. All 5 samples confirm standing?
   YES → Enter CONFIRMING state
   NO  → Return to IDLE (false positive filtered)
                    ↓
4. CONFIRMING state completes (1 second)
   → Emit "STAND_INITIATED" early signal
                    ↓
5. Enter MONITORING state (track BP drop)
                    ↓
6. Monitor for 10 seconds:
   - Track maximum SBP drop
   - Detect BP recovery (drop returns to 50% of peak)
                    ↓
7. Emit "STAND_CONFIRMED" event with full metrics
                    ↓
8. Enter REFRACTORY state (8 seconds)
   → Ignore all triggers during this period
                    ↓
9. Return to IDLE, ready for next event
```

### Blood Pressure Response Monitoring

During the MONITORING phase:
- **Maximum Drop**: Tracked continuously throughout the 10-second window
- **Recovery Time**: Time from stand initiation until BP drops to 50% of peak
- **Severity Classification**:
  - Mild: Drop < 15 mmHg
  - Moderate: 15-25 mmHg
  - Significant: 25-40 mmHg
  - Severe: > 40 mmHg

### Safety Features

1. **Debouncing**: 5-sample sliding window requires unanimous positive readings
2. **Refractory Period**: 8-second cooldown prevents double-counting
3. **SBP Smoothing**: 3-sample moving average reduces noise
4. **Activity Confirmation**: Sustained activity (5 seconds) required for device mode transitions
5. **Grace Period**: 3-second allowance for brief signal interruptions

---

## Installation

```bash
pip install -r requirements.txt
```

## Running the Dashboards

**Patient Dashboard:**
```bash
streamlit run patient_dashboard.py
```

**Clinical Dashboard:**
```bash
streamlit run clinical_dashboard.py
```

---

## Project Structure

```
bmeathon/
├── patient_dashboard.py      # Patient-friendly dashboard
├── clinical_dashboard.py      # Clinical monitoring dashboard
├── daily_stand_chart.py       # Chart rendering functions
├── state_machines.py          # State machine logic
├── advanced_features.py       # Posture detection (IMU)
├── compression_control.py     # Compression device control
├── data_generator.py          # Mock data generation
├── event_logger.py            # Event logging and persistence
├── event_logs/                # Stored event data (JSON)
└── src/
    └── data/
        ├── models.py          # Data structures
        ├── source.py          # Data source abstraction
        └── live/
            └── serial_source.py  # Arduino IMU sensor interface
```

---

## Hardware Integration

The system supports live Arduino IMU sensors via serial connection:

```python
from src.data.live.serial_source import SerialIMUSensor

sensor = SerialIMUSensor(port="/dev/tty.usbserial-DN04ABAX", baudrate=9600)
sensor.connect()
reading = sensor.read()  # Returns IMUSample with accel, gyro data
```

If no Arduino is connected, the system automatically falls back to mock data generation for development and testing.
