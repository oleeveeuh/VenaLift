"""
Patient/Caretaker Dashboard
Simple, comfort-focused interface with real-time Arduino stand detection
"""

import streamlit as st
import plotly.graph_objects as go
import time
from datetime import datetime, timedelta
from collections import deque
import numpy as np

# Import core components
from data_generator import PhysiologicalDataGenerator
from advanced_features import BiomechanicalFeatureExtractor, PostureState
from state_machines import DeviceModeStateMachine, DeviceMode, StandStateMachine, StandState
from compression_control import CompressionController, CompressionState
from event_logger import EventLogger
from daily_stand_chart import (
    render_compact_daily_stand_chart,
    render_weekly_activity_summary,
    render_longitudinal_mobility_trend,
    render_stand_activity_heatmap
)

# Arduino import
try:
    from src.data.live.serial_source import SerialIMUSensor
    ARDUINO_AVAILABLE = True
except ImportError:
    ARDUINO_AVAILABLE = False

# Professional CSS styling
PROFESSIONAL_CSS = """
<style>
    :root {
        --font-size: 16px;
        --background: linear-gradient(135deg, #e8f5f0 0%, #f0f9f4 50%, #e3f2f7 100%);
        --foreground: #1a3a3a;
        --card: rgba(255, 255, 255, 0.7);
        --card-foreground: #1a3a3a;
        --primary: #2d7a8f;
        --primary-foreground: #ffffff;
        --secondary: #4a9d6f;
        --secondary-foreground: #ffffff;
        --muted: #e6f4ed;
        --muted-foreground: #5a7c71;
        --border: rgba(45, 122, 143, 0.2);
        --radius: 0.75rem;
    }

    .stApp {
        background: var(--background);
    }

    .comfort-card {
        background: var(--card);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: var(--radius);
        box-shadow: 0 4px 12px rgba(45, 122, 143, 0.1);
        border: 1px solid var(--border);
        margin: 0.5rem 0;
        transition: transform 0.2s, box-shadow 0.2s;
    }

    .big-metric {
        font-size: 3.5rem !important;
        font-weight: 700;
        text-align: center;
        padding: 2rem;
        border-radius: var(--radius);
        background: linear-gradient(135deg, #2d7a8f 0%, #4a9d6f 100%);
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(45, 122, 143, 0.2);
    }

    .status-good {
        background: linear-gradient(135deg, #84fab0 0%, #4a9d6f 100%);
    }

    .status-warning {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
    }

    .status-info {
        background: linear-gradient(135deg, #a1c4fd 0%, #2d7a8f 100%);
    }

    .metric-container {
        background: var(--card);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: var(--radius);
        border: 1px solid var(--border);
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--primary);
    }

    .metric-label {
        font-size: 0.9rem;
        color: var(--muted-foreground);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.5rem;
    }

    .stButton > button {
        border-radius: var(--radius);
        font-size: 1rem;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        background: var(--primary);
        color: var(--primary-foreground);
        border: none;
        transition: all 0.2s;
    }

    .stButton > button:hover {
        background: var(--secondary);
        transform: translateY(-1px);
    }

    h1, h2, h3 {
        color: var(--foreground);
        font-weight: 600;
    }

    .stDeployButton {display: none;}
    footer {visibility: hidden;}

    .stInfo {
        background: var(--muted);
        border-left: 4px solid var(--primary);
        border-radius: var(--radius);
    }

    .stSuccess {
        background: #d4edda;
        border-left: 4px solid var(--secondary);
        border-radius: var(--radius);
    }

    .stWarning {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        border-radius: var(--radius);
    }

    .event-card {
        background: rgba(74, 157, 111, 0.15);
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        border-left: 3px solid #4a9d6f;
    }
</style>
"""


def read_arduino_data():
    """Read from Arduino and return formatted data"""
    if not st.session_state.get('arduino_connected', False):
        return None

    try:
        reading = st.session_state.arduino_sensor.read()

        if reading is None or reading.imu is None:
            return None

        imu = reading.imu
        ax, ay, az = imu.accel_x, imu.accel_y, imu.accel_z

        return {
            'timestamp': reading.timestamp,
            'accel_x': ax,
            'accel_y': ay,
            'accel_z': az,
            'gyro_x': 0.0,
            'gyro_y': 0.0,
            'gyro_z': 0.0,
            'posture': {'x': ax, 'y': ay, 'z': az},
            'blood_pressure': {
                'systolic': 120.0,
                'diastolic': 80.0,
                'hr': 72.0
            },
            'compression': st.session_state.compression_ctrl.current_pressure,
            'compression_state': 'Released',
            'data_source': 'arduino'
        }

    except Exception as e:
        st.session_state.arduino_connected = False
        return None


def render_header():
    """Render header"""
    st.markdown(PROFESSIONAL_CSS, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        st.title("Your Wellness Monitor")
        current_time = datetime.now().strftime("%I:%M %p")
        st.caption(f"{datetime.now().strftime('%A, %B %d')}  {current_time}")

    with col2:
        # Connection status
        if st.session_state.get('arduino_connected', False):
            st.success("Arduino Connected")
        elif ARDUINO_AVAILABLE:
            st.warning("Connecting...")
        else:
            st.info("Mock Data")

    with col3:
        if st.button("Reconnect"):
            reconnect_arduino()


def reconnect_arduino():
    """Reconnect to Arduino"""
    if st.session_state.get('arduino_sensor') is not None:
        try:
            st.session_state.arduino_sensor.disconnect()
        except:
            pass
        st.session_state.arduino_sensor = None
    st.session_state.arduino_connected = False

    if ARDUINO_AVAILABLE:
        try:
            st.session_state.arduino_sensor = SerialIMUSensor(
                port="/dev/tty.usbserial-DN04ABAX",
                baudrate=9600
            )
            st.session_state.arduino_sensor.connect()
            st.session_state.arduino_connected = True
            st.success("Reconnected!")
        except Exception as e:
            st.session_state.arduino_sensor = None
            st.session_state.arduino_connected = False
            st.error(f"Failed: {e}")

    st.rerun()


def render_comfort_status():
    """Large comfort status"""
    st.markdown("### How You're Doing")

    stand_count = st.session_state.stand_sm.stand_count

    # Generate encouraging status based on stand count
    if stand_count >= 10:
        status = "You're doing great today!"
        subtext = f"{stand_count} safe stands so far"
        gradient = "status-good"
    elif stand_count >= 5:
        status = "Nice progress today!"
        subtext = f"{stand_count} stands and counting"
        gradient = "status-good"
    elif stand_count >= 2:
        status = "You're building momentum!"
        subtext = f"{stand_count} stands completed"
        gradient = "status-info"
    else:
        status = "Every stand counts!"
        subtext = "Take your time, move at your own pace"
        gradient = "status-info"

    st.markdown(f"""
    <div class="big-metric {gradient}">
        <div style="font-size: 2rem;">{status}</div>
        <div style="font-size: 1.1rem; margin-top: 0.5rem; opacity: 0.9;">{subtext}</div>
    </div>
    """, unsafe_allow_html=True)


def render_activity_summary():
    """Activity summary - simplified to just times standing"""
    st.markdown("### Today's Activity")

    col1, col2 = st.columns(2)

    stand_count = st.session_state.stand_sm.stand_count

    with col1:
        daily_goal = 12
        progress = min(stand_count / daily_goal, 1.0) * 100

        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{stand_count}</div>
            <div class="metric-label">Times Stood Up Today</div>
            <div style="margin-top: 0.8rem;">
                <div style="background: #e6f4ed; border-radius: 10px; height: 10px; overflow: hidden;">
                    <div style="background: linear-gradient(90deg, #4a9d6f, #2d7a8f); height: 100%; width: {progress}%;"></div>
                </div>
                <div style="margin-top: 0.5rem; font-size: 0.85rem; color: #5a7c71;">Goal: {daily_goal}/day ({progress:.0f}%)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Safety reassurance card - check for risky events
        risky_count = 0
        for event in st.session_state.event_logger.event_buffer:
            if event.get('event_type') == 'stand_confirmed':
                if event.get('sbp_drop', 0) > 25:  # Significant drop threshold
                    risky_count += 1

        if risky_count == 0:
            safety_msg = "No risky transitions detected"
            safety_emoji = "✓"
            safety_color = "#4a9d6f"
        else:
            safety_msg = f"{risky_count} slow stands today"
            safety_emoji = "!"
            safety_color = "#fcb69f"

        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value" style="color: {safety_color};">{safety_emoji}</div>
            <div class="metric-label">Safety Status</div>
            <div style="margin-top: 0.8rem; font-size: 1rem; color: {safety_color}; font-weight: 600;">
                {safety_msg}
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_gentle_reminder():
    """Gentle coaching reminder based on current state"""
    st.markdown("### A Gentle Reminder")

    # Get current posture state
    current_posture = st.session_state.get('current_posture', 'Unknown')
    stability = st.session_state.bio_extractor.posture_stability_score \
        if hasattr(st.session_state.bio_extractor, 'posture_stability_score') else 0.7

    # Select appropriate reminder based on state
    import random

    reminders_when_sitting = [
        "Take a slow breath before standing",
        "When you're ready to stand, rise slowly",
        "Hydration helps with dizziness - stay hydrated!",
        "Move at your own pace, there's no rush"
    ]

    reminders_when_standing = [
        "You're doing great standing there!",
        "It might be a good time to sit and rest",
        "You've been steady - excellent balance",
        "Remember to breathe deeply while standing"
    ]

    if current_posture == "Standing":
        reminder = random.choice(reminders_when_standing)
        gradient = "status-good"
    else:
        reminder = random.choice(reminders_when_sitting)
        gradient = "status-info"

    st.markdown(f"""
    <div class="comfort-card {gradient}">
        <div style="text-align: center; font-size: 1.2rem; color: var(--card-foreground);">
            {reminder}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_daily_summary():
    """Simple daily summary card"""
    st.markdown("### Today's Highlights")

    stand_count = st.session_state.stand_sm.stand_count

    # Calculate stability trend (compare recent to older events)
    recent_stability = 0.8
    stability_trend = "Better than yesterday"

    # Get events from today for safe stands count
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    today_start_ts = today_start.timestamp()

    safe_stands = 0
    for event in st.session_state.event_logger.event_buffer:
        if event.get('event_type') == 'stand_confirmed':
            if event.get('timestamp', 0) >= today_start_ts:
                if event.get('sbp_drop', 0) < 25:  # Safe threshold
                    safe_stands += 1

    # Determine reminder based on progress
    if stand_count >= 10:
        reminder = "You're on track for a great day!"
    elif stand_count >= 5:
        reminder = "Keep up the steady pace"
    else:
        reminder = "Move slowly when standing"

    st.markdown(f"""
    <div class="comfort-card" style="background: rgba(74, 157, 111, 0.1);">
        <div style="display: flex; justify-content: space-around; text-align: center;">
            <div>
                <div style="font-size: 1.8rem; font-weight: 700; color: #2d7a8f;">{stand_count}</div>
                <div style="font-size: 0.85rem; color: #5a7c71;">Stands Today</div>
            </div>
            <div>
                <div style="font-size: 1.8rem; font-weight: 700; color: #4a9d6f;">{safe_stands}</div>
                <div style="font-size: 0.85rem; color: #5a7c71;">Safe Stands</div>
            </div>
            <div>
                <div style="font-size: 1.1rem; font-weight: 600; color: #4a9d6f;">{stability_trend}</div>
                <div style="font-size: 0.85rem; color: #5a7c71;">Stability</div>
            </div>
        </div>
        <div style="margin-top: 1rem; padding-top: 0.8rem; border-top: 1px solid rgba(74, 157, 111, 0.2); text-align: center; font-size: 0.95rem; color: #4a9d6f;">
            <strong>Tip:</strong> {reminder}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_realtime_status():
    """Recent stand events"""
    st.markdown("### Recent Stands")

    # Show recent stand events
    if 'recent_stand_events' in st.session_state and st.session_state.recent_stand_events:
        for event in st.session_state.recent_stand_events[:5]:
            st.markdown(f"""
            <div class="event-card">
                <strong>{event['time']}</strong>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.caption("No stands detected yet")


def render_therapy_status():
    """Current therapy status"""
    st.markdown("### Therapy Status")

    comp_status = st.session_state.compression_ctrl.get_status()

    if comp_status['is_engaged']:
        st.markdown("""
        <div class="comfort-card status-info">
            <h3 style="text-align: center; color: #ffffff;">Therapy Active</h3>
            <p style="text-align: center; font-size: 1.1rem; color: #ffffff;">
                You're receiving compression therapy. Relax and stay comfortable.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="comfort-card">
            <h3 style="text-align: center;">Therapy Resting</h3>
            <p style="text-align: center; font-size: 1.1rem; color: var(--muted-foreground);">
                Feel free to move around comfortably.
            </p>
        </div>
        """, unsafe_allow_html=True)


def render_encouragement():
    """Positive encouragement based on standing progress"""
    stand_count = st.session_state.stand_sm.stand_count

    if stand_count >= 12:
        message = "You've reached your standing goal! Amazing work today."
        msg_type = "success"
    elif stand_count >= 8:
        message = "Great progress! You're almost at your goal."
        msg_type = "success"
    elif stand_count >= 4:
        message = "You're building good momentum. Keep it up!"
        msg_type = "info"
    else:
        message = "Every stand is progress. Take your time!"
        msg_type = "info"

    if msg_type == "success":
        st.success(message)
    else:
        st.info(message)


def render_controls():
    """Simple control buttons"""
    st.markdown("### Quick Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Call for Help", use_container_width=True):
            st.warning("Calling caretaker... (Demo mode)")

    with col2:
        if st.button("Send Feedback", use_container_width=True):
            st.success("Feedback sent! (Demo mode)")

    with col3:
        if st.button("View Progress", use_container_width=True):
            st.info("Opening progress report... (Demo mode)")


class PatientDashboard:
    """Patient-friendly dashboard with real-time Arduino detection"""

    def __init__(self, update_interval_ms=300):
        self.update_interval = update_interval_ms / 1000.0

        if 'patient_initialized' not in st.session_state:
            self._initialize()

    def _initialize(self):
        """Initialize session state"""
        st.session_state.patient_initialized = True

        # Initialize Arduino connection status
        if 'arduino_connected' not in st.session_state:
            st.session_state.arduino_connected = False
        if 'arduino_sensor' not in st.session_state:
            st.session_state.arduino_sensor = None

        # Connect to Arduino
        if ARDUINO_AVAILABLE:
            try:
                st.session_state.arduino_sensor = SerialIMUSensor(
                    port="/dev/tty.usbserial-DN04ABAX",
                    baudrate=9600
                )
                st.session_state.arduino_sensor.connect()
                st.session_state.arduino_connected = True
            except Exception as e:
                st.session_state.arduino_sensor = None
                st.session_state.arduino_connected = False

        # Initialize components
        st.session_state.data_source = PhysiologicalDataGenerator(use_mock=True)
        st.session_state.bio_extractor = BiomechanicalFeatureExtractor(window_size=10)
        st.session_state.device_sm = DeviceModeStateMachine()
        st.session_state.compression_ctrl = CompressionController()
        st.session_state.stand_sm = StandStateMachine()
        st.session_state.event_logger = EventLogger(log_directory="./event_logs", buffer_size=200)

        st.session_state.start_time = time.time()
        st.session_state.is_running = True
        st.session_state.previous_stand_state = StandState.IDLE
        st.session_state.recent_stand_events = []
        st.session_state.last_reminder_update = 0  # Track when reminder was last updated

        # Load existing events from disk into buffer
        loaded = st.session_state.event_logger.load_events_from_disk(max_events=200)
        print(f"Loaded {loaded} events from disk")

        # Only generate mock data if no events were loaded
        if loaded == 0:
            self._generate_mock_stand_data()

        # Initialize stand count from today's logged events
        self._initialize_stand_count_from_logs()

        # Set mock therapy sessions if none exist
        if st.session_state.compression_ctrl.get_status()['cycle_count'] == 0:
            st.session_state.compression_ctrl.cycle_count = 6

    def update_data(self):
        """Main update loop with Arduino reading and stand detection"""
        # Try to read from Arduino
        raw_data = read_arduino_data()

        # Fall back to mock if Arduino not available
        if raw_data is None:
            raw_data = st.session_state.data_source.generate_sample()

        # Extract features
        bio_features = st.session_state.bio_extractor.extract(raw_data)

        # Update current posture state for display
        st.session_state.current_posture = bio_features['posture_state'].value
        st.session_state.current_pitch = bio_features['torso_pitch_angle']

        # Define IMU trigger (until IMU peak detector is wired in)
        imu_triggered = bio_features['posture_state'] == PostureState.STANDING

        # Update stand detection state machine
        stand_result = st.session_state.stand_sm.update(
            imu_stand_trigger=imu_triggered,
            sbp_drop=0.0,  # patient dashboard doesn't use BP
            timestamp=raw_data['timestamp']
        )

        # Detect stand event (same logic as clinical dashboard)
        current_stand_state = st.session_state.stand_sm.state
        previous_stand_state = st.session_state.get('previous_stand_state', StandState.IDLE)

        if stand_result is not None:

            # Log stand event
            from datetime import datetime as dt
            stand_timestamp = dt.now().strftime("%H:%M:%S")

            st.session_state.event_logger.log_stand_event(
                is_confirmed=True,
                timestamp=raw_data['timestamp'],
                pitch_angle=bio_features['torso_pitch_angle'],
                sbp_drop=0.0,
                posture_state=bio_features['posture_state'].value,
                bp_baseline=120.0,
                bp_current=120.0,
                recovery_time=0.0,
                max_drop=0.0
            )

            # Add to recent events
            st.session_state.recent_stand_events.insert(0, {
                'time': stand_timestamp,
                'timestamp': raw_data['timestamp'],
                'pitch': bio_features['torso_pitch_angle'],
                'sbp_drop': 0.0,
                'state': bio_features['posture_state'].value
            })
            # Keep only last 10
            if len(st.session_state.recent_stand_events) > 10:
                st.session_state.recent_stand_events.pop()

        st.session_state.previous_stand_state = current_stand_state

        # Update compression
        simulated_pressure = st.session_state.compression_ctrl.simulate_pressure_control()
        st.session_state.compression_ctrl.update(simulated_pressure, raw_data['timestamp'])

        return raw_data, bio_features

    def _generate_mock_stand_data(self):
        """Generate realistic mock stand events for visualization"""
        logger = st.session_state.event_logger
        now = datetime.now()

        # Realistic daily patterns for the past 7 days
        # Pattern: [6 days ago, 5 days ago, ..., today]
        daily_stand_counts = [8, 12, 6, 15, 9, 11, 7]  # Today has 7 stands

        for day_offset in range(7):
            target_date = (now - timedelta(days=6 - day_offset)).replace(hour=9, minute=0, second=0, microsecond=0)
            stands_today = daily_stand_counts[day_offset]

            # Generate stands spread throughout the day (8am - 8pm)
            for stand_idx in range(stands_today):
                hour = 8 + (stand_idx * (12 / stands_today))
                hour = min(hour, 20)
                minute = np.random.randint(0, 60)
                timestamp = target_date.replace(hour=int(hour), minute=minute, second=0).timestamp()

                # Most stands are safe (low BP drop), a few have higher drops
                if np.random.random() < 0.85:  # 85% safe stands
                    sbp_drop = float(np.random.randint(5, 20))  # Safe range
                else:
                    sbp_drop = float(np.random.randint(20, 35))  # Elevated but not dangerous

                bp_baseline = 115.0 + float(np.random.randint(-5, 15))
                bp_current = bp_baseline - sbp_drop

                logger.log_stand_event(
                    is_confirmed=True,
                    timestamp=timestamp,
                    pitch_angle=float(np.random.randint(65, 82)),
                    sbp_drop=sbp_drop,
                    posture_state="Standing",
                    bp_baseline=bp_baseline,
                    bp_current=bp_current
                )

    def _initialize_stand_count_from_logs(self):
        """Initialize stand count from today's logged events"""
        logger = st.session_state.event_logger
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        today_start_ts = today_start.timestamp()

        # Count events from today
        today_count = 0
        for event in logger.event_buffer:
            if event.get('event_type') == 'stand_confirmed':
                if event.get('timestamp', 0) >= today_start_ts:
                    today_count += 1

        # Set the stand count
        st.session_state.stand_sm.stand_count = today_count

    def run(self):
        """Main app"""
        st.set_page_config(
            page_title="Wellness Monitor",
            page_icon="",
            layout="wide",
            initial_sidebar_state="collapsed"
        )

        # Header
        render_header()

        st.divider()

        # Main status
        render_comfort_status()

        # Activity summary
        render_activity_summary()

        st.divider()

        # Daily summary card
        render_daily_summary()

        st.divider()

        # Gentle reminder
        gentle_reminder_placeholder = st.empty()
        with gentle_reminder_placeholder.container():
            render_gentle_reminder()

        st.divider()

        # Daily stand count chart (compact for patient view)
        st.markdown("### Recent Standing Activity")
        render_compact_daily_stand_chart(st.session_state.event_logger)

        st.divider()

        # Weekly activity summary
        render_weekly_activity_summary(st.session_state.event_logger)

        st.divider()

        # Standing activity pattern heatmap
        render_stand_activity_heatmap(st.session_state.event_logger)

        st.divider()

        # Longitudinal mobility trend
        render_longitudinal_mobility_trend(st.session_state.event_logger)

        st.divider()

        # Real-time status
        realtime_placeholder = st.empty()

        # Encouragement
        encouragement_placeholder = st.empty()

        st.divider()

        # Controls
        render_controls()

        # Main loop
        while True:
            if st.session_state.is_running:
                # Update data
                raw_data, bio_features = self.update_data()

                # Update gentle reminder only every 5 minutes (300 seconds)
                current_time = time.time()
                if current_time - st.session_state.last_reminder_update >= 300:
                    with gentle_reminder_placeholder.container():
                        render_gentle_reminder()
                    st.session_state.last_reminder_update = current_time

                # Render real-time status
                with realtime_placeholder.container():
                    render_realtime_status()

                # Render encouragement
                with encouragement_placeholder.container():
                    render_encouragement()

            # Wait
            time.sleep(self.update_interval)


if __name__ == "__main__":
    dashboard = PatientDashboard(update_interval_ms=300)
    dashboard.run()
