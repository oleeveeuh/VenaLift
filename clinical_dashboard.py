"""
Clinical Physiological Monitoring Dashboard
Essential features for clinical decision-making
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from collections import deque
import pandas as pd
import numpy as np
import json
from datetime import datetime

# Import all components
from data_generator import PhysiologicalDataGenerator
from advanced_features import (
    BiomechanicalFeatureExtractor,
    CardiovascularFeatureExtractor,
    CompressionFeatureExtractor,
    PostureState
)
from state_machines import (
    DeviceModeStateMachine,
    StandStateMachine,
    CalibrationStateMachine,
    DeviceMode,
    CalibrationState,
    StandEventData
)
from ml_module import (
    ContextualFeatureExtractor,
    StandPatternClassifier,
    MLAdjustmentEngine
)
from compression_control import (
    CompressionController,
    CompressionState,
    ReleaseRate
)
from event_logger import EventLogger, EventType
from daily_stand_chart import render_daily_stand_chart, render_stand_activity_heatmap, render_weekly_activity_summary

# Direct Arduino import
try:
    from src.data.live.serial_source import SerialIMUSensor
    ARDUINO_AVAILABLE = True
except ImportError:
    ARDUINO_AVAILABLE = False
    print("Warning: SerialIMUSensor not available, using mock data only")

# Professional CSS styling - matching original design
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

    /* Global styles */
    .stApp {
        background: var(--background);
    }

    /* Card styling */
    .dashboard-card {
        background: var(--card);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: var(--radius);
        box-shadow: 0 2px 8px rgba(45, 122, 143, 0.1);
        border: 1px solid var(--border);
        margin-bottom: 1rem;
    }

    /* Header styling */
    h1 {
        color: var(--foreground);
        font-weight: 600;
        font-size: 2rem;
    }

    h2, h3 {
        color: var(--foreground);
        font-weight: 600;
    }

    /* Metric cards */
    .metric-card {
        background: var(--card);
        backdrop-filter: blur(10px);
        padding: 1rem;
        border-radius: var(--radius);
        border: 1px solid var(--border);
        text-align: center;
    }

    /* Status indicators */
    .status-active {
        color: #4a9d6f;
        font-weight: 600;
    }

    .status-inactive {
        color: #fcb69f;
        font-weight: 600;
    }

    /* Button styling */
    .stButton > button {
        border-radius: var(--radius);
        font-size: 0.9rem;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.2s;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
    }

    /* Info boxes */
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

    .stError {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
        border-radius: var(--radius);
    }

    /* Hide streamlit elements */
    .stDeployButton {display: none;}
    footer {visibility: hidden;}
    footer:after {content: "";}

    /* Divider */
    hr {
        border-color: var(--border);
        opacity: 0.5;
    }
</style>
"""


class ClinicalDashboard:
    """
    Clinical dashboard with essential monitoring features
    """
    
    def __init__(self, update_interval_ms=300, buffer_size=100):
        """Initialize clinical dashboard"""
        self.update_interval = update_interval_ms / 1000.0
        self.buffer_size = buffer_size
        
        if 'initialized' not in st.session_state:
            self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize all session state variables"""
        st.session_state.initialized = True

        # Data source configuration
        if 'data_source_type' not in st.session_state:
            st.session_state.data_source_type = 'Serial'  # Default to Serial for Arduino

        # Initialize Arduino connection status
        if 'arduino_connected' not in st.session_state:
            st.session_state.arduino_connected = False
        if 'arduino_sensor' not in st.session_state:
            st.session_state.arduino_sensor = None

        # Connect to Arduino if Serial mode is selected
        if st.session_state.data_source_type == 'Serial' and ARDUINO_AVAILABLE:
            if not st.session_state.arduino_connected:
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
        else:
            st.session_state.arduino_sensor = None
            st.session_state.arduino_connected = False

        # Fallback data source
        st.session_state.data_source = PhysiologicalDataGenerator(use_mock=True)

        # Event logger - smaller buffer for better performance
        st.session_state.event_logger = EventLogger(
            log_directory="./event_logs",
            buffer_size=200  # Reduced from 500 for performance
        )

        # Load existing events from disk into buffer
        loaded = st.session_state.event_logger.load_events_from_disk(max_events=200)
        print(f"Loaded {loaded} events from disk")

        # Feature extractors
        st.session_state.bio_extractor = BiomechanicalFeatureExtractor(window_size=10)
        st.session_state.cv_extractor = CardiovascularFeatureExtractor(
            baseline_window=30, minimum_window=10
        )
        st.session_state.comp_extractor = CompressionFeatureExtractor()

        # State machines
        st.session_state.device_sm = DeviceModeStateMachine()
        st.session_state.stand_sm =StandStateMachine()
        st.session_state.cal_sm = CalibrationStateMachine()

        # ML components
        st.session_state.context_extractor = ContextualFeatureExtractor(window_duration=300)
        st.session_state.ml_classifier = StandPatternClassifier(use_trained_model=False)
        st.session_state.ml_adjuster = MLAdjustmentEngine()

        # Compression control
        st.session_state.compression_ctrl = CompressionController()

        # State
        st.session_state.is_running = True
        st.session_state.sample_count = 0
        st.session_state.start_time = time.time()

        # Auto-compression settings (enabled by default)
        st.session_state.auto_compression_enabled = True
        st.session_state.compression_target = 20.0
        st.session_state.compression_hold = 10.0
        st.session_state.compression_release_rate = ReleaseRate.MEDIUM

        # Recent stand events for display
        if 'recent_stand_events' not in st.session_state:
            st.session_state.recent_stand_events = []

        # Postural stability monitoring for standing instability detection
        if 'low_stability_start_time' not in st.session_state:
            st.session_state.low_stability_start_time = None
        if 'instability_alert_shown' not in st.session_state:
            st.session_state.instability_alert_shown = False

        # Cached today's stand count (updated periodically, not every loop)
        if 'today_stand_count' not in st.session_state:
            st.session_state.today_stand_count = 0
        if 'last_count_update' not in st.session_state:
            st.session_state.last_count_update = 0

        # Mobility baseline cache (updated every 10 seconds)
        if 'mobility_alert_cache' not in st.session_state:
            st.session_state.mobility_alert_cache = {'alert': None, 'last_check': 0}
        if 'orthostatic_alert_cache' not in st.session_state:
            st.session_state.orthostatic_alert_cache = {'alert': None, 'last_check': 0}

        # Minimal buffers - only essential data
        st.session_state.buffers = {
            'timestamps': deque(maxlen=self.buffer_size),
            'compression': deque(maxlen=self.buffer_size),
        }

        # Load existing events from disk into buffer
        loaded = st.session_state.event_logger.load_events_from_disk(max_events=200)

        # Only generate mock data if no events were loaded from disk
        if loaded == 0:
            self._generate_mock_stands(num_events=150)
            print(f"Generated mock data (no files found on disk)")
        else:
            print(f"Loaded {loaded} events from disk, skipping mock generation")

        # Start with compression released (not engaged) on initialization
        # Daily dose and cycle count start at 0 for the new session
        st.session_state.compression_ctrl.daily_compression_dose = 0.0
        st.session_state.compression_ctrl.cycle_count = 0
    
    def _read_arduino_direct(self):
        if not st.session_state.get('arduino_connected', False):
            return None

        try:
            reading = st.session_state.arduino_sensor.read()

            # This is NORMAL
            if reading is None or reading.imu is None:
                return None

            imu = reading.imu
            ax, ay, az = imu.accel_x, imu.accel_y, imu.accel_z

            # Generate realistic BP variation by default
            sbp = np.clip(np.random.normal(120, 5), 105, 140)
            dbp = sbp * 0.6 + np.random.normal(0, 2)
            hr = np.clip(np.random.normal(72, 5), 55, 100)

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
                    'systolic': sbp,
                    'diastolic': dbp,
                    'hr': hr
                },
                'compression': 0.0,
                'compression_state': 'Released',
                'data_source': 'arduino'
            }

        except Exception as e:
            print(f"Arduino error: {e}")
            st.session_state.arduino_connected = False
            return None


    def update_data(self):
        """Main update loop"""
        # Get raw data - use direct Arduino reader if connected, otherwise mock
        raw_data = self._read_arduino_direct()

        # Fallback to mock data source if Arduino not connected
        if raw_data is None and hasattr(st.session_state, 'data_source'):
            raw_data = st.session_state.data_source.generate_sample()

        # Safety: guard against missing Arduino data
        if raw_data is None:
            return None, None, None, None, None, None, None

        # Update compression controller
        prev_cycle_count = st.session_state.compression_ctrl.cycle_count
        simulated_pressure = st.session_state.compression_ctrl.simulate_pressure_control()
        st.session_state.compression_ctrl.update(simulated_pressure, raw_data['timestamp'])
        raw_data['compression'] = st.session_state.compression_ctrl.current_pressure

        # Detect compression cycle completion and log it
        if st.session_state.compression_ctrl.cycle_count > prev_cycle_count:
            # A compression cycle just completed
            ctrl = st.session_state.compression_ctrl
            st.session_state.event_logger.log_compression_cycle(
                timestamp=raw_data['timestamp'],
                target_pressure=ctrl.target_pressure,
                achieved_pressure=ctrl.target_pressure * 0.95,  # Approximate achieved
                hold_duration=ctrl.hold_duration,
                release_rate=ctrl.release_rate.value,
                cycle_dose=ctrl.target_pressure * ctrl.hold_duration if ctrl.hold_duration > 0 else 200.0,
                was_auto=True
            )

        # Extract features
        bio_features = st.session_state.bio_extractor.extract(raw_data)
        cv_features = st.session_state.cv_extractor.extract(raw_data, bio_features['posture_state'])
        comp_features = st.session_state.comp_extractor.extract(raw_data)

        raw_data['pitch_angle'] = bio_features['torso_pitch_angle']

        # Update state machines
        has_activity = bio_features['posture_state'].value in ("Walking", "Standing")
        st.session_state.device_sm.update(has_activity=has_activity)

        # Define IMU trigger (until IMU peak detector is wired in)
        imu_triggered = bio_features['posture_state'].value == "Standing"

        stand_result = st.session_state.stand_sm.update(
            imu_stand_trigger=imu_triggered,
            sbp_drop=cv_features['sbp_drop'],
            timestamp=raw_data['timestamp']
        )

        # EARLY: stand just started (during motion)
        if stand_result == "STAND_INITIATED":
            # UI update - log early event
            st.session_state.event_logger.log_event(
                EventType.STAND_INITIATED,
                timestamp=raw_data['timestamp'],
                pitch_angle=bio_features['torso_pitch_angle'],
                posture_state=bio_features['posture_state'].value
            )

            # Auto-compression: engage full cycle on stand initiation
            if st.session_state.auto_compression_enabled:
                st.session_state.compression_ctrl.engage_compression(
                    st.session_state.compression_target,
                    st.session_state.compression_hold,
                    st.session_state.compression_release_rate
                )

        # LATE: stand fully completed (after monitoring)
        elif isinstance(stand_result, StandEventData):
            from datetime import datetime as dt
            stand_timestamp = dt.now().strftime("%H:%M:%S")

            # Log confirmed stand event
            st.session_state.event_logger.log_stand_event(
                is_confirmed=True,
                timestamp=raw_data['timestamp'],
                pitch_angle=bio_features['torso_pitch_angle'],
                sbp_drop=cv_features['sbp_drop'],
                posture_state=bio_features['posture_state'].value,
                bp_baseline=cv_features.get('sbp_baseline', 120.0),
                bp_current=cv_features.get('sbp_baseline', 120.0) - cv_features['sbp_drop'],
                recovery_time=stand_result.recovery_time or 0.0,
                max_drop=stand_result.max_sbp_drop
            )

            # Auto-compression: engage when stand is confirmed
            # This provides compression therapy immediately after standing is detected
            if st.session_state.auto_compression_enabled:
                st.session_state.compression_ctrl.engage_compression(
                    st.session_state.compression_target,
                    st.session_state.compression_hold,
                    st.session_state.compression_release_rate
                )
            from datetime import datetime as dt
            stand_timestamp = dt.now().strftime("%H:%M:%S")

            # Log confirmed stand event
            st.session_state.event_logger.log_stand_event(
                is_confirmed=True,
                timestamp=raw_data['timestamp'],
                pitch_angle=bio_features['torso_pitch_angle'],
                sbp_drop=cv_features['sbp_drop'],
                posture_state=bio_features['posture_state'].value,
                bp_baseline=cv_features.get('sbp_baseline', 120.0),
                bp_current=cv_features.get('sbp_baseline', 120.0) - cv_features['sbp_drop'],
                recovery_time=stand_result.recovery_time or 0.0,
                max_drop=stand_result.max_sbp_drop
            )

            # Add to recent events
            if 'recent_stand_events' not in st.session_state:
                st.session_state.recent_stand_events = []
            st.session_state.recent_stand_events.insert(0, {
                'time': stand_timestamp,
                'timestamp': raw_data['timestamp'],
                'pitch': bio_features['torso_pitch_angle'],
                'sbp_drop': cv_features['sbp_drop'],
                'state': bio_features['posture_state'].value
            })

            # Keep only last 10 events
            st.session_state.recent_stand_events = st.session_state.recent_stand_events[:10]

        # Update ML
        st.session_state.context_extractor.update_bp_drop(
            cv_features['sbp_drop'],
            raw_data['timestamp']
        )
        contextual_features = st.session_state.context_extractor.extract(raw_data['timestamp'])
        ml_prediction = st.session_state.ml_classifier.predict(contextual_features)
        adjusted_params = st.session_state.ml_adjuster.adjust_parameters(ml_prediction)
        
        # Update buffers
        elapsed = time.time() - st.session_state.start_time
        st.session_state.buffers['timestamps'].append(elapsed)
        st.session_state.buffers['compression'].append(st.session_state.compression_ctrl.current_pressure)
        
        st.session_state.sample_count += 1
        
        return (raw_data, bio_features, cv_features, comp_features,
                contextual_features, ml_prediction, adjusted_params)
    
    def render_header(self):
        """Render dashboard header"""
        st.markdown(PROFESSIONAL_CSS, unsafe_allow_html=True)
        st.title("Clinical Physiological Monitoring")

        # Create placeholders for dynamic metrics (updated in main loop)
        if 'header_metrics_placeholders' not in st.session_state:
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                session_time_placeholder = st.empty()
            with col2:
                confirmed_stands_placeholder = st.empty()
            with col3:
                compression_cycles_placeholder = st.empty()
            with col4:
                data_source_placeholder = st.empty()
            with col5:
                status_placeholder = st.empty()

            st.session_state.header_metrics_placeholders = {
                'session_time': session_time_placeholder,
                'confirmed_stands': confirmed_stands_placeholder,
                'compression_cycles': compression_cycles_placeholder,
                'data_source': data_source_placeholder,
                'status': status_placeholder
            }

        # Initial render
        self._update_header_metrics()

    def _update_header_metrics(self):
        """Update the dynamic header metrics"""
        if 'header_metrics_placeholders' not in st.session_state:
            return

        placeholders = st.session_state.header_metrics_placeholders
        metrics = st.session_state.event_logger.get_daily_metrics()
        comp_status = st.session_state.compression_ctrl.get_status()
        arduino_connected = st.session_state.get('arduino_connected', False)
        ds_type = st.session_state.get('data_source_type', 'Serial')

        # Session Time
        elapsed = time.time() - st.session_state.start_time
        placeholders['session_time'].metric("Session Time", f"{elapsed/60:.1f} min")

        # Confirmed Stands (today only - cached for performance)
        # Only recalculate every 5 seconds or when explicitly needed
        current_time = time.time()
        if (current_time - st.session_state.last_count_update > 5.0 or
            st.session_state.today_stand_count == 0):
            # Recalculate today's stand count
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            today_start_ts = today_start.timestamp()
            st.session_state.today_stand_count = sum(
                1 for e in st.session_state.event_logger.event_buffer
                if e.get('event_type') == 'stand_confirmed'
                and e.get('timestamp', 0) >= today_start_ts
            )
            st.session_state.last_count_update = current_time

        placeholders['confirmed_stands'].metric("Confirmed Stands", st.session_state.today_stand_count)

        # Compression Cycles
        placeholders['compression_cycles'].metric("Compression Cycles", comp_status['cycle_count'])

        # Data Source Status
        if ds_type == "Serial":
            if arduino_connected:
                placeholders['data_source'].metric("Data Source", "Arduino IMU")
            else:
                if not ARDUINO_AVAILABLE:
                    placeholders['data_source'].metric("Data Source", "Mock", help="Arduino module not found")
                else:
                    placeholders['data_source'].metric("Data Source", "Connecting...")
        else:
            placeholders['data_source'].metric("Data Source", "Mock")

        # Status
        if st.session_state.is_running:
            placeholders['status'].metric("Status", "Active")
        else:
            placeholders['status'].metric("Status", "Paused")

    def render_controls(self):
        """Render control buttons"""
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            if st.button("Pause" if st.session_state.is_running else "Resume"):
                st.session_state.is_running = not st.session_state.is_running

        with col2:
            if st.button("Reset Session"):
                st.session_state.sample_count = 0
                st.session_state.start_time = time.time()
                st.session_state.recent_stand_events = []
                for buffer in st.session_state.buffers.values():
                    buffer.clear()
                # Also reset compression controller
                st.session_state.compression_ctrl.cycle_count = 0
                st.session_state.compression_ctrl.daily_compression_dose = 0.0
                st.rerun()

        with col3:
            if st.button("Engage Compression"):
                st.session_state.compression_ctrl.engage_compression(
                    st.session_state.compression_target,
                    st.session_state.compression_hold,
                    st.session_state.compression_release_rate
                )

        with col4:
            if st.button("Release Compression"):
                st.session_state.compression_ctrl.release_compression()

        with col5:
            if st.button("Generate Mock Data"):
                # Reset cycle count before generating
                st.session_state.compression_ctrl.cycle_count = 0
                st.session_state.compression_ctrl.daily_compression_dose = 0.0
                self._generate_mock_stands(num_events=150)
                st.success("Generated 150 mock stand events across 5 months")
                st.rerun()

        # Clinical Alert Testing Tools
        with st.expander("🧪 Test Clinical Alerts"):
            st.markdown("**Generate mock data to trigger specific clinical alerts:**")

            alert_col1, alert_col2, alert_col3, alert_col4 = st.columns(4)

            with alert_col1:
                if st.button("🔴 Orthostatic BP", key="test_orthostatic"):
                    self._generate_orthostatic_drop_alert()
                    st.success("Generated mock data for Orthostatic BP Drop alert")
                    st.rerun()

            with alert_col2:
                if st.button("⚠️ Reduced Mobility", key="test_mobility"):
                    self._generate_mobility_alert()
                    st.success("Generated mock data for Reduced Mobility alert")
                    st.rerun()

            with alert_col3:
                if st.button("🚶 Instability", key="test_instability"):
                    self._generate_instability_alert()
                    st.success("Simulated standing instability - alert will appear")
                    st.rerun()

            with alert_col4:
                if st.button("🔄 Reset", key="test_reset"):
                    # Reset cycle count before generating
                    st.session_state.compression_ctrl.cycle_count = 0
                    st.session_state.compression_ctrl.daily_compression_dose = 0.0
                    # Reset instability tracking
                    st.session_state.low_stability_start_time = None
                    st.session_state.instability_alert_shown = False
                    self._generate_mock_stands(num_events=150)
                    st.success("Reset to normal mock data")
                    st.rerun()

    def _add_mock_compression_data(self, cycles=5):
        """Add mock compression cycle data to controller and event logger"""
        ctrl = st.session_state.compression_ctrl
        logger = st.session_state.event_logger
        now = time.time()

        # Reset to realistic baseline
        ctrl.daily_compression_dose = 0.0

        for i in range(cycles):
            # Directly update controller stats
            ctrl.cycle_count += 1
            ctrl.last_cycle_time = now - (cycles - i) * 300  # 5 min apart

            # Realistic values:
            # - Target pressure: 15-25 kPa (typical compression therapy range)
            # - Hold duration: 8-15 seconds per cycle
            # - Dose per cycle: pressure * duration ≈ 150-375 kPa·s
            target_p = 18.0 + np.random.uniform(-3, 7)  # 15-25 kPa
            hold_duration = 10.0 + np.random.uniform(-2, 5)  # 8-15 sec
            achieved_p = target_p * (0.92 + np.random.uniform(-0.02, 0.05))  # Slight variation
            cycle_dose = achieved_p * hold_duration  # Realistic dose calculation

            ctrl.daily_compression_dose += cycle_dose

            # Log compression cycle
            logger.log_compression_cycle(
                timestamp=now - (cycles - i) * 300 + 10,
                target_pressure=target_p,
                achieved_pressure=achieved_p,
                hold_duration=hold_duration,
                release_rate="Medium",
                cycle_dose=cycle_dose,
                was_auto=True
            )

    def _reconnect_arduino(self):
        """Reconnect to Arduino"""
        # Disconnect existing
        if st.session_state.get('arduino_sensor') is not None:
            try:
                st.session_state.arduino_sensor.disconnect()
            except:
                pass
            st.session_state.arduino_sensor = None
        st.session_state.arduino_connected = False

        # Try to reconnect
        if ARDUINO_AVAILABLE:
            try:
                st.session_state.arduino_sensor = SerialIMUSensor(
                    port="/dev/tty.usbserial-DN04ABAX",
                    baudrate=9600
                )
                st.session_state.arduino_sensor.connect()
                st.session_state.arduino_connected = True
                st.session_state.data_source_type = 'Serial'
                st.success("Reconnected to Arduino")
            except Exception as e:
                st.session_state.arduino_sensor = None
                st.session_state.arduino_connected = False
                st.session_state.data_source_type = 'Mock'
                st.error(f"Arduino connection failed: {e}")
                # this thing is constantly udpating 
        else:
            st.error("Arduino module not available")

        st.rerun()

    def _generate_mock_stands(self, num_events=150):
        """Generate mock stand events spanning multiple months for visualization"""
        from datetime import datetime, timedelta
        logger = st.session_state.event_logger
        ctrl = st.session_state.compression_ctrl
        now = datetime.now()

        # Generate events spread across 5 months
        months_to_generate = 5
        events_per_month = num_events // months_to_generate  # ~30 events per month

        for month_offset in range(months_to_generate):
            # Calculate target month
            target_year = now.year
            target_month = now.month - month_offset
            while target_month <= 0:
                target_month += 12
                target_year -= 1

            # Get days in this month
            if target_month in [1, 3, 5, 7, 8, 10, 12]:
                days_in_month = 31
            elif target_month in [4, 6, 9, 11]:
                days_in_month = 30
            else:
                # Handle February (account for leap years)
                if (target_year % 4 == 0 and target_year % 100 != 0) or (target_year % 400 == 0):
                    days_in_month = 29
                else:
                    days_in_month = 28

            # Generate events for this month
            if month_offset == 0:
                # For current month, generate events across last 7 days only
                # This ensures the "Stands per Day" chart shows meaningful data
                # NOTE: No compression cycles added for current month to keep "Cycles Today" at 0
                for day_offset in range(7):
                    target_date = (now - timedelta(days=6 - day_offset)).replace(hour=10, minute=0, second=0)

                    # More varied stands per day (8-18 stands) for realism
                    # Each day has a different pattern: some active, some less active
                    daily_patterns = [12, 15, 8, 18, 10, 14, 9]  # Varied daily counts
                    stands_today = daily_patterns[day_offset]

                    for stand_idx in range(stands_today):
                        # Spread stands throughout the day (8am to 8pm)
                        hour = 8 + (stand_idx * (12 / stands_today))
                        hour = min(hour, 20)  # Cap at 8pm
                        minute = np.random.randint(0, 60)
                        timestamp = target_date.replace(hour=int(hour), minute=minute, second=0).timestamp()

                        # Generate realistic BP and posture data for this stand
                        is_confirmed = np.random.random() > 0.1
                        sbp_drop = np.clip(np.random.normal(16, 4), 5, 35)
                        recovery_time = np.clip(sbp_drop * 0.8 + np.random.normal(2, 1), 3, 30)
                        pitch_angle = np.clip(np.random.normal(70, 10), 45, 90) if is_confirmed else np.random.normal(40, 5)

                        logger.log_stand_event(
                            is_confirmed=is_confirmed,
                            timestamp=timestamp,
                            pitch_angle=pitch_angle,
                            sbp_drop=sbp_drop,
                            posture_state="Standing" if is_confirmed else "Seated",
                            bp_baseline=120.0 + np.random.normal(0, 3),
                            bp_current=120.0 - sbp_drop,
                            recovery_time=recovery_time,
                            max_drop=sbp_drop + np.random.uniform(0, 3)
                        )

                        # Note: No compression cycles added for current month data
                        # This keeps "Cycles Today" starting at 0
            else:
                # For historical months, distribute events across days of the month
                for event_idx in range(events_per_month):
                    day_of_month = (event_idx % days_in_month) + 1

                    try:
                        target_date = datetime(target_year, target_month, day_of_month)
                        weekday = target_date.weekday()

                        # Set different times based on day of week
                        if weekday < 3:  # Mon-Wed: morning peaks (8-11am)
                            hour = 8 + (event_idx % 4)
                        elif weekday < 5:  # Thu-Fri: afternoon peaks (1-5pm)
                            hour = 13 + (event_idx % 5)
                        else:  # Sat-Sun: midday (10am-3pm)
                            hour = 10 + (event_idx % 5)

                        # Clamp hour to valid range
                        hour = min(hour, 17)
                        minute = np.random.randint(0, 60)
                        stand_time = target_date.replace(hour=hour, minute=minute, second=0)
                        timestamp = stand_time.timestamp()
                    except ValueError:
                        continue

                    # Generate realistic BP and posture data
                    is_confirmed = np.random.random() > 0.1
                    sbp_drop = np.clip(np.random.normal(16, 4), 5, 35)
                    recovery_time = np.clip(sbp_drop * 0.8 + np.random.normal(2, 1), 3, 30)
                    pitch_angle = np.clip(np.random.normal(70, 10), 45, 90) if is_confirmed else np.random.normal(40, 5)

                    logger.log_stand_event(
                        is_confirmed=is_confirmed,
                        timestamp=timestamp,
                        pitch_angle=pitch_angle,
                        sbp_drop=sbp_drop,
                        posture_state="Standing" if is_confirmed else "Seated",
                        bp_baseline=120.0 + np.random.normal(0, 3),
                        bp_current=120.0 - sbp_drop,
                        recovery_time=recovery_time,
                        max_drop=sbp_drop + np.random.uniform(0, 3)
                    )

                    # Log compression cycle for all CONFIRMED stands (not random)
                    # This ensures compression cycles match confirmed stands
                    if is_confirmed:
                        target_p = 18.0 + np.random.uniform(-3, 7)  # 15-25 kPa
                        hold_duration = 10.0 + np.random.uniform(-2, 5)  # 8-15 sec
                        achieved_p = target_p * (0.92 + np.random.uniform(-0.02, 0.05))
                        cycle_dose = achieved_p * hold_duration

                        ctrl.daily_compression_dose += cycle_dose

                        logger.log_compression_cycle(
                            timestamp=timestamp + 10,
                            target_pressure=target_p,
                            achieved_pressure=achieved_p,
                            hold_duration=hold_duration,
                            release_rate="Medium",
                            cycle_dose=cycle_dose,
                            was_auto=True
                        )

    def _generate_orthostatic_drop_alert(self):
        """Generate mock data to trigger orthostatic BP drop alert"""
        from event_logger import EventLogger
        from datetime import datetime, timedelta
        now = datetime.now()

        # Create a NEW event logger with a larger buffer
        logger = EventLogger(log_directory="./event_logs", buffer_size=500)
        st.session_state.event_logger = logger

        # Generate normal baseline data for past 13 days (good mobility)
        for day_offset in range(13, 0, -1):
            target_date = (now - timedelta(days=day_offset)).replace(hour=10, minute=0, second=0)
            for i in range(20):  # 20 stands per day = normal
                timestamp = (target_date + timedelta(minutes=i*30)).timestamp()
                logger.log_stand_event(
                    is_confirmed=True,
                    timestamp=timestamp,
                    pitch_angle=70.0,
                    sbp_drop=12.0,  # Normal drop
                    posture_state="Standing",
                    bp_baseline=120.0,
                    bp_current=108.0
                )

        # Today: generate multiple stands with significant orthostatic drops (≥20 mmHg)
        for i in range(3):
            hours_ago = (2 - i)  # Spread across today
            timestamp = (now - timedelta(hours=hours_ago, minutes=i*15)).timestamp()

            # Generate orthostatic drop (25-32 mmHg)
            sbp_drop = 25.0 + i * 3.5
            bp_baseline = 125.0
            bp_current = bp_baseline - sbp_drop

            logger.log_stand_event(
                is_confirmed=True,
                timestamp=timestamp,
                pitch_angle=68.0 + i * 2,
                sbp_drop=sbp_drop,
                posture_state="Standing",
                bp_baseline=bp_baseline,
                bp_current=bp_current,
                recovery_time=15.0 + i * 5,
                max_drop=sbp_drop + 2.0
            )

    def _generate_mobility_alert(self):
        """Generate mock data to trigger reduced mobility alert"""
        from event_logger import EventLogger
        from datetime import datetime, timedelta
        now = datetime.now()

        # Create a NEW event logger with a larger buffer to hold all the test data
        # This avoids the maxlen=50 issue with the existing logger
        logger = EventLogger(log_directory="./event_logs", buffer_size=500)
        st.session_state.event_logger = logger

        # Generate normal baseline data for past 13 days (high mobility)
        for day_offset in range(13, 0, -1):
            target_date = (now - timedelta(days=day_offset)).replace(hour=10, minute=0, second=0)
            for i in range(25):  # 25 stands per day = high baseline
                timestamp = (target_date + timedelta(minutes=i*20)).timestamp()
                logger.log_stand_event(
                    is_confirmed=True,
                    timestamp=timestamp,
                    pitch_angle=72.0,
                    sbp_drop=14.0,
                    posture_state="Standing",
                    bp_baseline=118.0,
                    bp_current=104.0
                )

        # Today: generate very few stands (<70% of baseline)
        # Baseline avg = 25, so we need <17.5 stands. Let's do 8 stands.
        today_stands = 8  # ~32% of baseline = 68% decrease

        for i in range(today_stands):
            hours_ago = (7 - i)  # Spread across today
            timestamp = (now - timedelta(hours=hours_ago)).timestamp()

            logger.log_stand_event(
                is_confirmed=True,
                timestamp=timestamp,
                pitch_angle=70.0,
                sbp_drop=15.0,
                posture_state="Standing",
                bp_baseline=120.0,
                bp_current=105.0
            )

    def _generate_instability_alert(self):
        """Simulate standing instability by setting low stability state"""
        import time

        # Set low stability start time to 35 seconds ago to trigger alert immediately
        st.session_state.low_stability_start_time = time.time() - 35
        st.session_state.instability_alert_shown = False

    def render_clinical_alerts(self, cv_features, bio_features):
        """Render critical clinical alerts dashboard"""
        st.subheader("Clinical Alerts")
        
        alerts = []
        
        # Alert 1: Orthostatic hypotension (SBP drop >20 mmHg)
        if cv_features['sbp_drop'] > 20:
            alerts.append({
                'severity': 'HIGH',
                'type': 'Orthostatic Hypotension',
                'message': f'SBP ↓{cv_features["sbp_drop"]:.0f} mmHg',
                'action': 'Review standing protocol'
            })
        
        # Alert 2: Severe drop (>30 mmHg)
        if cv_features['sbp_drop'] > 30:
            alerts.append({
                'severity': 'CRITICAL',
                'type': 'Severe BP Drop',
                'message': f'SBP ↓{cv_features["sbp_drop"]:.0f} mmHg',
                'action': 'IMMEDIATE attention required'
            })
        
        # Alert 3: Frequent severe drops
        recent_events = st.session_state.event_logger.get_recent_events(n=50)
        recent_severe = [e for e in recent_events 
                        if e.get('event_type') == 'stand_confirmed' 
                        and e.get('bp_response', {}).get('sbp_drop', 0) > 20]
        
        if len(recent_severe) >= 3:
            alerts.append({
                'severity': 'MEDIUM',
                'type': 'Fall Risk',
                'message': f'{len(recent_severe)} severe drops in recent stands',
                'action': 'Consider compression adjustment'
            })
        
        # Alert 4: Poor posture stability
        if bio_features['posture_stability_score'] < 0.3:
            alerts.append({
                'severity': 'MEDIUM',
                'type': 'Instability Detected',
                'message': f'Stability score: {bio_features["posture_stability_score"]:.2f}',
                'action': 'Monitor patient closely'
            })
        
        # Alert 5: High daily compression dose
        comp_status = st.session_state.compression_ctrl.get_status()
        if comp_status['daily_dose'] > 10000:
            alerts.append({
                'severity': 'MEDIUM',
                'type': 'High Compression Dose',
                'message': f'Daily dose: {comp_status["daily_dose"]:.0f} kPa·s',
                'action': 'Review compression schedule'
            })

        # Alert 6: Reduced standing frequency (14-day rolling baseline comparison)
        # Cached for performance - only check every 10 seconds
        current_time = time.time()
        if current_time - st.session_state.mobility_alert_cache['last_check'] > 10:
            st.session_state.mobility_alert_cache['alert'] = self._check_mobility_baseline()
            st.session_state.mobility_alert_cache['last_check'] = current_time
        mobility_alert = st.session_state.mobility_alert_cache['alert']
        if mobility_alert:
            alerts.append(mobility_alert)

        # Alert 7: Daily orthostatic BP drops (≥20 mmHg within 3 min of standing)
        # Cached for performance - only check every 5 seconds
        if current_time - st.session_state.orthostatic_alert_cache['last_check'] > 5:
            st.session_state.orthostatic_alert_cache['alert'] = self._check_daily_orthostatic_drops()
            st.session_state.orthostatic_alert_cache['last_check'] = current_time
        orthostatic_alert = st.session_state.orthostatic_alert_cache['alert']
        if orthostatic_alert:
            alerts.append(orthostatic_alert)

        # Alert 8: Standing instability (low posture stability for >30 seconds during standing)
        instability_alert = self._check_standing_instability(bio_features)
        if instability_alert:
            alerts.append(instability_alert)

        # Display alerts
        if alerts:
            from datetime import datetime
            alert_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            for alert in alerts:
                # Determine styling based on severity
                if alert['severity'] == 'CRITICAL':
                    st.error(f"**{alert['type']}:** {alert['message']}")
                    st.caption(f"-> {alert['action']}")
                elif alert['severity'] == 'HIGH':
                    st.error(f"**{alert['type']}:** {alert['message']}")
                    st.caption(f"-> {alert['action']}")
                elif alert['severity'] == 'MEDIUM':
                    st.warning(f"**{alert['type']}:** {alert['message']}")
                    st.caption(f"-> {alert['action']}")

                # Show timestamp
                st.caption(f"📅 {alert_time}")

                # Show low confidence warning if applicable
                if alert.get('low_confidence', False):
                    st.caption("⚠️ Interpret with caution – limited recent data")
                st.caption("")  # Spacing between alerts
        else:
            st.success("No active alerts - All metrics within normal ranges")

    def _check_standing_instability(self, bio_features):
        """
        Check for prolonged low postural stability during standing.
        Triggers an alert if stability remains below 0.5 for more than 30 seconds while standing.

        Args:
            bio_features: Dictionary containing biomechanical features including posture_stability_score

        Returns:
            dict with alert details if instability detected, None otherwise
        """
        import time

        stability_score = bio_features.get('posture_stability_score', 1.0)
        current_time = time.time()
        instability_threshold = 0.5
        instability_duration_threshold = 30  # seconds

        # Get current posture state - handle both enum and string
        posture_state_obj = bio_features.get('posture_state', 'Unknown')
        if hasattr(posture_state_obj, 'value'):
            posture_state = posture_state_obj.value
        else:
            posture_state = posture_state_obj

        # Only check during standing
        if posture_state == 'Standing':
            if stability_score < instability_threshold:
                # Low stability detected
                if st.session_state.low_stability_start_time is None:
                    # Start tracking low stability period
                    st.session_state.low_stability_start_time = current_time
                else:
                    # Check how long stability has been low
                    low_stability_duration = current_time - st.session_state.low_stability_start_time

                    if low_stability_duration > instability_duration_threshold:
                        # Has been unstable for more than threshold - trigger alert
                        if not st.session_state.instability_alert_shown:
                            st.session_state.instability_alert_shown = True
                            return {
                                'severity': 'HIGH',
                                'type': '⚠️ Standing Instability Detected',
                                'message': 'Low postural stability observed during standing',
                                'action': f'Stability below {instability_threshold:.1f} for {int(low_stability_duration)}s'
                            }
            else:
                # Stability is good - reset tracking
                st.session_state.low_stability_start_time = None
                st.session_state.instability_alert_shown = False
        else:
            # Not standing - reset tracking
            st.session_state.low_stability_start_time = None
            st.session_state.instability_alert_shown = False

        return None

    def _check_mobility_baseline(self):
        """
        Check today's standing frequency against 14-day rolling baseline.

        Returns:
            dict with alert details if mobility is reduced, None otherwise
        """
        from datetime import datetime, timedelta
        import numpy as np

        logger = st.session_state.event_logger
        now = datetime.now()

        # Get today's start time
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        today_start_ts = today_start.timestamp()

        # Get events from the last 14 days (including today)
        fourteen_days_ago = (now - timedelta(days=14)).timestamp()

        # Count stands by day
        daily_counts = {}
        all_events = logger.get_recent_events(n=200)

        for event in all_events:
            if event.get('event_type') == 'stand_confirmed':
                ts = event.get('timestamp', 0)
                if ts >= fourteen_days_ago:
                    # Get the date for this event
                    event_date = datetime.fromtimestamp(ts).date()
                    if event_date not in daily_counts:
                        daily_counts[event_date] = 0
                    daily_counts[event_date] += 1

        # Need at least 7 days of data for meaningful baseline
        if len(daily_counts) < 7:
            return None

        # Calculate today's count
        today_date = now.date()
        today_count = daily_counts.get(today_date, 0)

        # Calculate 14-day rolling average (excluding today for baseline)
        baseline_days = [date for date in daily_counts.keys() if date != today_date]
        if not baseline_days:
            return None

        baseline_counts = [daily_counts[date] for date in baseline_days]
        baseline_avg = np.mean(baseline_counts)

        # Avoid division by zero
        if baseline_avg < 1:
            return None

        # Check if today's count is less than 70% of baseline
        ratio = today_count / baseline_avg
        if ratio < 0.7:
            percent_decrease = int((1 - ratio) * 100)
            return {
                'severity': 'MEDIUM',
                'type': '⚠️ Reduced Mobility Detected',
                'message': f'Standing frequency decreased by {percent_decrease}% compared to baseline',
                'action': f'{today_count} stands today vs {baseline_avg:.0f} baseline average'
            }

        return None

    def _check_daily_orthostatic_drops(self):
        """
        Check for orthostatic blood pressure drops (≥20 mmHg) within the past 24 hours.

        Detects standing transitions where systolic BP drops by 20 mmHg or more
        within 3 minutes of standing.

        Returns:
            dict with alert details if orthostatic drops detected, None otherwise
        """
        from datetime import datetime, timedelta

        logger = st.session_state.event_logger
        now = datetime.now()

        # Get events from the past 24 hours
        twenty_four_hours_ago = (now - timedelta(hours=24)).timestamp()

        # Get all stand events from the past 24 hours
        all_events = logger.get_recent_events(n=200)
        recent_stands = []

        for event in all_events:
            if event.get('event_type') == 'stand_confirmed':
                ts = event.get('timestamp', 0)
                if ts >= twenty_four_hours_ago:
                    sbp_drop = event.get('bp_response', {}).get('sbp_drop', 0)
                    if sbp_drop >= 20:  # Orthostatic threshold
                        recent_stands.append({
                            'timestamp': ts,
                            'sbp_drop': sbp_drop,
                            'time_str': datetime.fromtimestamp(ts).strftime('%H:%M')
                        })

        # Trigger alert if one or more orthostatic drops detected today
        if recent_stands:
            # Get the most significant drop (maximum)
            max_drop_event = max(recent_stands, key=lambda x: x['sbp_drop'])
            drop_value = int(max_drop_event['sbp_drop'])

            # Severity based on drop magnitude
            if drop_value >= 30:
                severity = 'HIGH'
                icon = '🔴'
            else:
                severity = 'MEDIUM'
                icon = '🟠'

            count_msg = f"{len(recent_stands)} time{'s' if len(recent_stands) > 1 else ''}"
            if len(recent_stands) > 1:
                action = f"{count_msg} today, max: {drop_value} mmHg at {max_drop_event['time_str']}"
            else:
                action = f"Detected at {max_drop_event['time_str']}"

            return {
                'severity': severity,
                'type': f'{icon} Orthostatic BP Drop Detected',
                'message': f'Systolic blood pressure dropped {drop_value} mmHg following standing',
                'action': action
            }

        return None

    def render_stand_quality_metrics(self):
        """Render clinical stand assessment metrics"""
        st.subheader("Stand Quality Metrics")
        
        metrics = st.session_state.event_logger.get_daily_metrics()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_drop = metrics['bp_metrics'].get('avg_drop', 0)
            st.metric(
                "Avg BP Drop",
                f"{avg_drop:.1f} mmHg",
                delta=None,
                delta_color="inverse",
                help="Average blood pressure drop during stands"
            )
            
            if avg_drop == 0:
                st.caption("No data yet")
            elif avg_drop < 10:
                st.caption("Excellent")
            elif avg_drop < 20:
                st.caption("Good")
            else:
                st.caption("Needs attention")
        
        with col2:
            confirmed = metrics['stand_events']['confirmed']
            total = metrics['stand_events']['total_attempts']
            
            if total > 0:
                accuracy = (confirmed / total) * 100
                st.metric(
                    "Detection Accuracy",
                    f"{accuracy:.0f}%",
                    help="Valid stands / total attempts"
                )
                
                if accuracy > 90:
                    st.caption("Excellent")
                elif accuracy > 75:
                    st.caption("Good")
                else:
                    st.caption("Review calibration")
            else:
                st.metric("Detection Accuracy", "N/A")
                st.caption("No stands yet")
        
        with col3:
            max_drop = metrics['bp_metrics'].get('max_drop', 0)
            st.metric(
                "Max BP Drop",
                f"{max_drop:.1f} mmHg",
                help="Maximum blood pressure drop today"
            )
            
            if max_drop == 0:
                st.caption("No data yet")
            elif max_drop > 30:
                st.caption("Critical level")
            elif max_drop > 20:
                st.caption("High")
            else:
                st.caption("Within range")
        
        with col4:
            cycles = metrics['compression_metrics']['cycles']
            st.metric(
                "Compression Cycles",
                cycles,
                help="Therapy sessions completed today"
            )
            
            target = 12
            if cycles == 0:
                st.caption("No therapy yet")
            elif cycles >= target:
                st.caption("Target met")
            else:
                remaining = target - cycles
                st.caption(f"{remaining} more needed")
    
    def render_compression_efficacy(self):
        """Show compression therapy effectiveness comparison"""
        st.subheader("Compression Therapy Efficacy")

        recent_events = st.session_state.event_logger.get_recent_events(n=100)
        stand_events = [e for e in recent_events if e.get('event_type') == 'stand_confirmed']

        # Separate compressed vs uncompressed stands
        compressed_stands = []
        uncompressed_stands = []

        if len(stand_events) >= 2:
            compression_events = [e for e in recent_events if e.get('event_type') == 'compression_cycle']
            compression_times = [e['timestamp'] for e in compression_events]

            for stand in stand_events:
                stand_time = stand['timestamp']
                bp_drop = stand.get('bp_response', {}).get('sbp_drop', 0)

                # Check if compression was active within 60s before stand
                was_compressed = any(abs(stand_time - comp_time) < 60 for comp_time in compression_times)

                if was_compressed:
                    compressed_stands.append(bp_drop)
                else:
                    uncompressed_stands.append(bp_drop)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Without Compression**")
            if uncompressed_stands:
                avg_uncomp = np.mean(uncompressed_stands)
                max_uncomp = np.max(uncompressed_stands)
                st.metric("Avg BP Drop", f"{avg_uncomp:.1f} mmHg")
                st.metric("Max Drop", f"{max_uncomp:.1f} mmHg")
                st.caption(f"n={len(uncompressed_stands)} stands")
            else:
                st.caption("No uncompressed stands")

        with col2:
            st.markdown("**With Compression**")
            if compressed_stands:
                avg_comp = np.mean(compressed_stands)
                max_comp = np.max(compressed_stands)
                st.metric("Avg BP Drop", f"{avg_comp:.1f} mmHg")
                st.metric("Max Drop", f"{max_comp:.1f} mmHg")
                st.caption(f"n={len(compressed_stands)} stands")
            else:
                st.caption("No compressed stands")

        with col3:
            st.markdown("**Improvement**")
            if compressed_stands and uncompressed_stands:
                improvement = np.mean(uncompressed_stands) - np.mean(compressed_stands)
                improvement_pct = (improvement / np.mean(uncompressed_stands)) * 100

                st.metric(
                    "Benefit",
                    f"{improvement:.1f} mmHg",
                    delta=f"{improvement_pct:.0f}%"
                )

                if improvement > 5:
                    st.success("Therapy effective")
                elif improvement > 0:
                    st.info("Modest benefit")
                else:
                    st.warning("Review therapy")
            else:
                st.caption("Need both types")

        # Always render the chart to avoid duplicate element ID issues
        fig = go.Figure()

        if compressed_stands and uncompressed_stands:
            fig.add_trace(go.Box(
                y=uncompressed_stands,
                name='No Compression',
                marker_color='#fcb69f',
                boxmean='sd'
            ))

            fig.add_trace(go.Box(
                y=compressed_stands,
                name='With Compression',
                marker_color='#4a9d6f',
                boxmean='sd'
            ))
        else:
            # Show empty chart when no data
            fig.add_trace(go.Box(
                y=[0],
                name='No Data',
                marker_color='#e0e0e0'
            ))
            fig.update_annotations([
                dict(text="Need more stand events with compression data",
                     xref="paper", yref="paper",
                     x=0.5, y=0.5, showarrow=False)
            ])

        fig.update_layout(
            yaxis_title="SBP Drop (mmHg)",
            height=300,
            showlegend=True
        )

        # Use dynamic key based on data hash to avoid duplicate ID issues
        data_hash = hash((len(compressed_stands), len(uncompressed_stands), time.time()))
        st.plotly_chart(fig, use_container_width=True, key=f"compression_efficacy_{data_hash}")
    
    def render_quick_export(self):
        """One-click export for EMR/clinical records"""
        st.subheader("Clinical Export")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            if st.button("Generate Summary", use_container_width=True):
                st.session_state.generate_summary = True
        
        if st.session_state.get('generate_summary', False):
            metrics = st.session_state.event_logger.get_daily_metrics()
            
            summary = {
                'report_type': 'Physiological Monitoring Session Summary',
                'generated': datetime.now().isoformat(),
                'session_info': {
                    'duration_hours': (time.time() - st.session_state.start_time) / 3600,
                    'samples_collected': st.session_state.sample_count
                },
                'stand_events': {
                    'total_confirmed': metrics['stand_events']['confirmed'],
                    'false_positives': metrics['stand_events']['false_positive'],
                    'detection_accuracy': f"{(metrics['stand_events']['confirmed'] / max(1, metrics['stand_events']['total_attempts'])) * 100:.1f}%"
                },
                'blood_pressure': {
                    'avg_drop_mmHg': metrics['bp_metrics'].get('avg_drop', 0),
                    'max_drop_mmHg': metrics['bp_metrics'].get('max_drop', 0),
                    'min_drop_mmHg': metrics['bp_metrics'].get('min_drop', 0),
                    'total_measurements': metrics['bp_metrics']['drops_recorded']
                },
                'compression_therapy': {
                    'cycles_completed': metrics['compression_metrics']['cycles'],
                    'total_dose_kPa_s': metrics['compression_metrics']['total_dose'],
                    'avg_pressure_kPa': metrics['compression_metrics'].get('avg_pressure', 0),
                    'avg_hold_time_s': metrics['compression_metrics'].get('avg_hold_time', 0)
                },
                'clinical_recommendations': self._generate_recommendations(metrics)
            }
            
            # JSON download
            col1.download_button(
                "Download JSON",
                json.dumps(summary, indent=2),
                f"clinical_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
            
            # Formatted text
            formatted_text = self._format_clinical_text(summary)
            
            with col1:
                st.text_area(
                    "Copy to Medical Record:",
                    formatted_text,
                    height=300
                )
    
    def _generate_recommendations(self, metrics):
        """Generate clinical recommendations"""
        recommendations = []
        
        avg_drop = metrics['bp_metrics'].get('avg_drop', 0)
        max_drop = metrics['bp_metrics'].get('max_drop', 0)
        cycles = metrics['compression_metrics']['cycles']
        
        if avg_drop > 20:
            recommendations.append("HIGH: Average BP drop >20 mmHg - Consider increasing compression therapy frequency")
        elif avg_drop > 15:
            recommendations.append("MODERATE: BP drops elevated - Monitor closely")
        else:
            recommendations.append("GOOD: BP drops within acceptable range")
        
        target_cycles = 12
        if cycles < target_cycles * 0.5:
            recommendations.append("CONCERN: Low therapy adherence - Counsel patient")
        elif cycles < target_cycles * 0.8:
            recommendations.append("MODERATE: Below target cycles - Encourage compliance")
        else:
            recommendations.append("GOOD: Therapy compliance on target")
        
        if max_drop > 30:
            recommendations.append("URGENT: Severe orthostatic hypotension - Review protocol")
        
        return recommendations
    
    def _format_clinical_text(self, summary):
        """Format summary as clinical text"""
        text = f"""PHYSIOLOGICAL MONITORING SESSION SUMMARY
Generated: {datetime.fromisoformat(summary['generated']).strftime('%Y-%m-%d %H:%M')}

SESSION INFORMATION:
- Duration: {summary['session_info']['duration_hours']:.2f} hours
- Samples: {summary['session_info']['samples_collected']}

STAND EVENTS:
- Confirmed Stands: {summary['stand_events']['total_confirmed']}
- False Positives: {summary['stand_events']['false_positives']}
- Detection Accuracy: {summary['stand_events']['detection_accuracy']}

BLOOD PRESSURE RESPONSE:
- Average Drop: {summary['blood_pressure']['avg_drop_mmHg']:.1f} mmHg
- Maximum Drop: {summary['blood_pressure']['max_drop_mmHg']:.1f} mmHg
- Minimum Drop: {summary['blood_pressure']['min_drop_mmHg']:.1f} mmHg

COMPRESSION THERAPY:
- Cycles Completed: {summary['compression_therapy']['cycles_completed']}
- Total Dose: {summary['compression_therapy']['total_dose_kPa_s']:.0f} kPa·s
- Average Pressure: {summary['compression_therapy']['avg_pressure_kPa']:.1f} kPa

CLINICAL RECOMMENDATIONS:
"""
        for i, rec in enumerate(summary['clinical_recommendations'], 1):
            text += f"{i}. {rec}\n"
        
        return text
    
    def render_compression_summary(self):
        """Simplified compression status displayed prominently at top of dashboard"""
        st.markdown("### 💨 Compression Therapy")

        status = st.session_state.compression_ctrl.get_status()

        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Status", status['state'])
            if status['is_engaged']:
                st.success("Active")
            else:
                st.info("Released")
        
        with col2:
            st.metric("Current Pressure", f"{status['current_pressure']:.1f} kPa")
            pressure_pct = (status['current_pressure'] / 30.0)
            st.progress(min(pressure_pct, 1.0))
        
        with col3:
            st.metric("Daily Dose", f"{status['daily_dose']:.0f} kPa·s")
            if status['daily_dose'] > 5000:
                st.warning("High dose")
        
        with col4:
            st.metric("Cycles Today", status['cycle_count'])

    def render_realtime_events(self, bio_features, cv_features):
        """Render real-time event log and current state"""
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Current State")

            # Current posture state
            posture_state = bio_features['posture_state'].value
            pitch = bio_features['torso_pitch_angle']
            stability = bio_features['posture_stability_score']

            st.metric("Posture", posture_state)
            st.metric("Pitch Angle", f"{pitch:.1f}°")
            st.metric("Stability", f"{stability:.2f}")

            # Stand detection state
            stand_state = st.session_state.stand_sm.state.value
            st.metric("Stand Detection", stand_state)

        with col2:
            st.subheader("Recent Stand Events")

            # Show recent stand events
            if 'recent_stand_events' not in st.session_state:
                st.session_state.recent_stand_events = []

            if st.session_state.recent_stand_events:
                for event in st.session_state.recent_stand_events[:5]:  # Show last 5
                    st.markdown(f"""
                    <div style="background: rgba(74, 157, 111, 0.1); padding: 0.5rem;
                                border-radius: 0.5rem; margin-bottom: 0.5rem; border-left: 3px solid #4a9d6f;">
                        <strong>{event['time']}</strong> | Pitch: {event['pitch']:.1f}° | BP Drop: {event['sbp_drop']:.1f} mmHg
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No stand events detected yet")

    def run(self):
        """Main application loop"""
        st.set_page_config(
            page_title="Clinical Monitoring Dashboard",
            page_icon="",
            layout="wide"
        )
        
        # Header
        self.render_header()
        self.render_controls()
        
        st.divider()
        
        # Auto-compression settings
        with st.expander("Auto-Compression Settings"):
            col1, col2 = st.columns(2)

            with col1:
                auto_enabled = st.checkbox(
                    "Enable Auto-Compression",
                    value=st.session_state.auto_compression_enabled
                )
                st.session_state.auto_compression_enabled = auto_enabled

                target = st.slider(
                    "Target Pressure (kPa)",
                    5.0, 30.0, st.session_state.compression_target, 1.0
                )
                st.session_state.compression_target = target

            with col2:
                hold = st.slider(
                    "Hold Duration (s)",
                    1.0, 30.0, st.session_state.compression_hold, 1.0
                )
                st.session_state.compression_hold = hold

                release = st.selectbox(
                    "Release Rate",
                    [ReleaseRate.SLOW, ReleaseRate.MEDIUM, ReleaseRate.FAST],
                    index=1,
                    format_func=lambda x: x.value
                )
                st.session_state.compression_release_rate = release

        # Data source toggle
        col1, col2, col3 = st.columns([2, 2, 4])
        with col1:
            current_source = st.session_state.get('data_source_type', 'Serial')
            new_source = st.radio("Data Source", ["Serial (Arduino)", "Mock"], index=0 if current_source == "Serial" else 1, horizontal=True)
        with col2:
            if st.button("Apply Data Source"):
                target_source = "Serial" if "Serial" in new_source else "Mock"
                if target_source != current_source:
                    # Disconnect existing
                    if st.session_state.get('arduino_sensor') is not None:
                        try:
                            st.session_state.arduino_sensor.disconnect()
                        except:
                            pass
                    st.session_state.arduino_sensor = None
                    st.session_state.arduino_connected = False

                    # Connect if Serial
                    if target_source == "Serial" and ARDUINO_AVAILABLE:
                        try:
                            st.session_state.arduino_sensor = SerialIMUSensor(
                                port="/dev/tty.usbserial-DN04ABAX",
                                baudrate=9600
                            )
                            st.session_state.arduino_sensor.connect()
                            st.session_state.arduino_connected = True
                            st.success("Switched to Arduino")
                        except Exception as e:
                            st.error(f"Connection failed: {e}")
                            st.session_state.data_source_type = 'Mock'
                    else:
                        st.session_state.data_source_type = 'Mock'
                        st.success("Switched to Mock data")

                    st.rerun()
        with col3:
            # Show Arduino availability
            if ARDUINO_AVAILABLE:
                st.info("Arduino module available")
            else:
                st.warning("Arduino module not found")

        st.divider()

        # Compression therapy status placeholder (updated in real-time)
        compression_summary_placeholder = st.empty()

        st.divider()

        # Real-time events placeholder
        events_placeholder = st.empty()

        st.divider()

        # Clinical alerts placeholder
        alerts_placeholder = st.empty()

        st.divider()

        # Stand quality metrics placeholder
        quality_metrics_placeholder = st.empty()

        st.divider()

        # Compression efficacy placeholder
        efficacy_placeholder = st.empty()

        st.divider()
        
        # Quick export (static)
        self.render_quick_export()

        st.divider()

        # Daily stand count chart
        render_daily_stand_chart(st.session_state.event_logger)

        st.divider()

        # Standing activity pattern heatmap
        render_stand_activity_heatmap(st.session_state.event_logger)

        st.divider()

        # Weekly activity summary
        render_weekly_activity_summary(st.session_state.event_logger)

        # Main loop
        while True:
            if st.session_state.is_running:
                # Update data
                result = self.update_data()

                # Skip rendering if no data available
                if result is None or result[0] is None:
                    time.sleep(self.update_interval)
                    continue

                (raw_data, bio_features, cv_features, comp_features,
                 contextual_features, ml_prediction, adjusted_params) = result

                # Render real-time events (updates immediately on stand detection)
                with events_placeholder.container():
                    self.render_realtime_events(bio_features, cv_features)

                # Render clinical alerts (real-time)
                with alerts_placeholder.container():
                    self.render_clinical_alerts(cv_features, bio_features)

                # Update header metrics (real-time)
                self._update_header_metrics()

                # Render compression summary (real-time)
                with compression_summary_placeholder.container():
                    self.render_compression_summary()

                # Render stand quality metrics (update periodically)
                with quality_metrics_placeholder.container():
                    self.render_stand_quality_metrics()

                # Render compression efficacy (update periodically)
                with efficacy_placeholder.container():
                    self.render_compression_efficacy()

            # Wait
            time.sleep(self.update_interval)


if __name__ == "__main__":
    dashboard = ClinicalDashboard(update_interval_ms=300, buffer_size=100)
    dashboard.run()
