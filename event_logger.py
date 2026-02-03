"""
Event Logging Module
Comprehensive logging system for physiological monitoring events
"""

import json
import os
from collections import deque
from datetime import datetime, date
from typing import Dict, List, Optional, Any
from enum import Enum
import numpy as np


class EventType(Enum):
    """Event type classifications"""
    STAND_INITIATED = "stand_initiated"
    STAND_CONFIRMED = "stand_confirmed"
    STAND_FALSE_POSITIVE = "stand_false_positive"
    BP_RESPONSE = "bp_response"
    COMPRESSION_CYCLE = "compression_cycle"
    USER_FEEDBACK = "user_feedback"
    SYSTEM_ALERT = "system_alert"


class EventLogger:
    """
    Manages event logging with JSON output and aggregation
    
    Features:
    - Per-event JSON records
    - Daily aggregation
    - In-memory event buffer
    - Disk persistence
    """
    
    def __init__(self, log_directory: str = "./event_logs", buffer_size: int = 100):
        """
        Initialize event logger
        
        Args:
            log_directory: Directory for JSON log files
            buffer_size: Number of recent events to keep in memory
        """
        self.log_directory = log_directory
        self.buffer_size = buffer_size
        
        # Create log directory if it doesn't exist
        os.makedirs(log_directory, exist_ok=True)
        
        # In-memory event buffer (for dashboard display)
        self.event_buffer = deque(maxlen=buffer_size)
        
        # Daily aggregated metrics
        self.daily_metrics = {
            'date': date.today().isoformat(),
            'stand_events': {
                'confirmed': 0,
                'false_positive': 0,
                'total_attempts': 0
            },
            'bp_metrics': {
                'drops_recorded': 0,
                'avg_drop': 0.0,
                'max_drop': 0.0,
                'min_drop': 0.0,
                'drops_list': []
            },
            'compression_metrics': {
                'cycles': 0,
                'total_dose': 0.0,
                'avg_pressure': 0.0,
                'avg_hold_time': 0.0,
                'pressures': [],
                'hold_times': []
            },
            'user_feedback': {
                'count': 0,
                'ratings': [],
                'comments': []
            },
            'system_alerts': {
                'count': 0,
                'types': {}
            }
        }
        
        # Event ID counter
        self.event_counter = 0
        
        # Current date for daily reset
        self.current_date = date.today()
    
    def log_stand_event(self, 
                       is_confirmed: bool,
                       timestamp: float,
                       pitch_angle: float,
                       sbp_drop: float,
                       posture_state: str,
                       bp_baseline: float,
                       bp_current: float,
                       recovery_time: Optional[float] = None,
                       max_drop: Optional[float] = None,
                       **kwargs) -> Dict:
        """
        Log a stand event (confirmed or false-positive)
        
        Args:
            is_confirmed: True if valid stand, False if false-positive
            timestamp: Event timestamp
            pitch_angle: Torso pitch at stand (degrees)
            sbp_drop: SBP drop magnitude (mmHg)
            posture_state: Posture classification
            bp_baseline: Baseline SBP (mmHg)
            bp_current: Current SBP (mmHg)
            recovery_time: Time to BP recovery (seconds)
            max_drop: Maximum SBP drop during event (mmHg)
            **kwargs: Additional parameters
            
        Returns:
            Event record dictionary
        """
        event_type = EventType.STAND_CONFIRMED if is_confirmed else EventType.STAND_FALSE_POSITIVE
        
        event = {
            'event_id': self._get_next_id(),
            'event_type': event_type.value,
            'timestamp': timestamp,
            'datetime': datetime.fromtimestamp(timestamp).isoformat(),
            'is_confirmed': is_confirmed,
            'biomechanics': {
                'pitch_angle': round(pitch_angle, 2),
                'posture_state': posture_state
            },
            'bp_response': {
                'baseline_sbp': round(bp_baseline, 1),
                'current_sbp': round(bp_current, 1),
                'sbp_drop': round(sbp_drop, 1),
                'max_drop': round(max_drop, 1) if max_drop else round(sbp_drop, 1),
                'recovery_time': round(recovery_time, 2) if recovery_time else None
            },
            'additional_data': kwargs
        }
        
        # Add to buffer and save
        self._add_event(event)

        # Update daily metrics
        self._update_stand_metrics(is_confirmed, sbp_drop)
        # Also update BP metrics for drops_list
        if is_confirmed:
            self._update_bp_metrics(sbp_drop)

        return event
    
    def log_bp_response(self,
                       timestamp: float,
                       sbp_drop: float,
                       sbp_baseline: float,
                       sbp_minimum: float,
                       post_stand_slope: float,
                       context: str = "",
                       **kwargs) -> Dict:
        """
        Log a blood pressure response event
        
        Args:
            timestamp: Event timestamp
            sbp_drop: SBP drop magnitude (mmHg)
            sbp_baseline: Baseline SBP (mmHg)
            sbp_minimum: Minimum SBP reached (mmHg)
            post_stand_slope: BP recovery slope (mmHg/s)
            context: Context description
            **kwargs: Additional parameters
            
        Returns:
            Event record dictionary
        """
        event = {
            'event_id': self._get_next_id(),
            'event_type': EventType.BP_RESPONSE.value,
            'timestamp': timestamp,
            'datetime': datetime.fromtimestamp(timestamp).isoformat(),
            'bp_metrics': {
                'sbp_baseline': round(sbp_baseline, 1),
                'sbp_minimum': round(sbp_minimum, 1),
                'sbp_drop': round(sbp_drop, 1),
                'post_stand_slope': round(post_stand_slope, 3)
            },
            'context': context,
            'additional_data': kwargs
        }
        
        self._add_event(event)
        self._update_bp_metrics(sbp_drop)
        
        return event
    
    def log_compression_cycle(self,
                             timestamp: float,
                             target_pressure: float,
                             achieved_pressure: float,
                             hold_duration: float,
                             release_rate: str,
                             cycle_dose: float,
                             was_auto: bool = False,
                             **kwargs) -> Dict:
        """
        Log a compression cycle event
        
        Args:
            timestamp: Event timestamp
            target_pressure: Target pressure (kPa)
            achieved_pressure: Actual pressure achieved (kPa)
            hold_duration: Hold time (seconds)
            release_rate: Release rate setting
            cycle_dose: Compression dose for this cycle (kPa·s)
            was_auto: True if auto-triggered
            **kwargs: Additional parameters
            
        Returns:
            Event record dictionary
        """
        event = {
            'event_id': self._get_next_id(),
            'event_type': EventType.COMPRESSION_CYCLE.value,
            'timestamp': timestamp,
            'datetime': datetime.fromtimestamp(timestamp).isoformat(),
            'compression_params': {
                'target_pressure': round(target_pressure, 1),
                'achieved_pressure': round(achieved_pressure, 1),
                'hold_duration': round(hold_duration, 2),
                'release_rate': release_rate,
                'cycle_dose': round(cycle_dose, 1),
                'auto_triggered': was_auto
            },
            'additional_data': kwargs
        }
        
        self._add_event(event)
        self._update_compression_metrics(achieved_pressure, hold_duration, cycle_dose)
        
        return event
    
    def log_user_feedback(self,
                         timestamp: float,
                         feedback_type: str,
                         rating: Optional[int] = None,
                         comment: str = "",
                         related_event_id: Optional[int] = None,
                         **kwargs) -> Dict:
        """
        Log user feedback event
        
        Args:
            timestamp: Event timestamp
            feedback_type: Type of feedback
            rating: Numerical rating (1-5)
            comment: Text comment
            related_event_id: ID of related event
            **kwargs: Additional parameters
            
        Returns:
            Event record dictionary
        """
        event = {
            'event_id': self._get_next_id(),
            'event_type': EventType.USER_FEEDBACK.value,
            'timestamp': timestamp,
            'datetime': datetime.fromtimestamp(timestamp).isoformat(),
            'feedback': {
                'type': feedback_type,
                'rating': rating,
                'comment': comment,
                'related_event_id': related_event_id
            },
            'additional_data': kwargs
        }
        
        self._add_event(event)
        self._update_feedback_metrics(rating, comment)
        
        return event
    
    def log_system_alert(self,
                        timestamp: float,
                        alert_type: str,
                        severity: str,
                        message: str,
                        **kwargs) -> Dict:
        """
        Log system alert event
        
        Args:
            timestamp: Event timestamp
            alert_type: Type of alert
            severity: Alert severity (info/warning/error)
            message: Alert message
            **kwargs: Additional parameters
            
        Returns:
            Event record dictionary
        """
        event = {
            'event_id': self._get_next_id(),
            'event_type': EventType.SYSTEM_ALERT.value,
            'timestamp': timestamp,
            'datetime': datetime.fromtimestamp(timestamp).isoformat(),
            'alert': {
                'type': alert_type,
                'severity': severity,
                'message': message
            },
            'additional_data': kwargs
        }
        
        self._add_event(event)
        self._update_alert_metrics(alert_type)
        
        return event

    def log_event(self, event_type: EventType, timestamp: float, **kwargs) -> Dict:
        """
        Log a generic event by type

        Args:
            event_type: EventType enum value
            timestamp: Event timestamp
            **kwargs: Additional event-specific data

        Returns:
            Event record dictionary
        """
        event = {
            'event_id': self._get_next_id(),
            'event_type': event_type.value,
            'timestamp': timestamp,
            'datetime': datetime.fromtimestamp(timestamp).isoformat(),
            'data': kwargs
        }

        self._add_event(event)

        return event

    def _add_event(self, event: Dict) -> None:
        """
        Add event to buffer and save to disk
        
        Args:
            event: Event dictionary
        """
        # Check if need to reset daily metrics
        self._check_daily_reset()
        
        # Add to in-memory buffer
        self.event_buffer.append(event)
        
        # Save to disk
        self._save_event_to_disk(event)
    
    def _save_event_to_disk(self, event: Dict) -> None:
        """
        Save event as JSON file
        
        Args:
            event: Event dictionary
        """
        # Create filename with timestamp and event ID
        timestamp = datetime.fromtimestamp(event['timestamp'])
        filename = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{event['event_id']}_{event['event_type']}.json"
        filepath = os.path.join(self.log_directory, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(event, f, indent=2)
        except Exception as e:
            print(f"Error saving event to disk: {e}")
    
    def _get_next_id(self) -> int:
        """Get next event ID"""
        self.event_counter += 1
        return self.event_counter
    
    def _check_daily_reset(self) -> None:
        """Check if need to reset daily metrics"""
        today = date.today()
        
        if today != self.current_date:
            # Save previous day's metrics
            self._save_daily_metrics()
            
            # Reset metrics
            self._reset_daily_metrics()
            self.current_date = today
    
    def _reset_daily_metrics(self) -> None:
        """Reset daily metrics to initial state"""
        self.daily_metrics = {
            'date': date.today().isoformat(),
            'stand_events': {
                'confirmed': 0,
                'false_positive': 0,
                'total_attempts': 0
            },
            'bp_metrics': {
                'drops_recorded': 0,
                'avg_drop': 0.0,
                'max_drop': 0.0,
                'min_drop': 0.0,
                'drops_list': []
            },
            'compression_metrics': {
                'cycles': 0,
                'total_dose': 0.0,
                'avg_pressure': 0.0,
                'avg_hold_time': 0.0,
                'pressures': [],
                'hold_times': []
            },
            'user_feedback': {
                'count': 0,
                'ratings': [],
                'comments': []
            },
            'system_alerts': {
                'count': 0,
                'types': {}
            }
        }
    
    def _save_daily_metrics(self) -> None:
        """Save daily metrics to disk"""
        filename = f"daily_metrics_{self.daily_metrics['date']}.json"
        filepath = os.path.join(self.log_directory, filename)
        
        # Calculate final averages
        metrics = self.daily_metrics.copy()
        
        if metrics['bp_metrics']['drops_list']:
            metrics['bp_metrics']['avg_drop'] = np.mean(metrics['bp_metrics']['drops_list'])
            metrics['bp_metrics']['max_drop'] = np.max(metrics['bp_metrics']['drops_list'])
            metrics['bp_metrics']['min_drop'] = np.min(metrics['bp_metrics']['drops_list'])
            del metrics['bp_metrics']['drops_list']  # Don't save full list
        
        if metrics['compression_metrics']['pressures']:
            metrics['compression_metrics']['avg_pressure'] = np.mean(
                metrics['compression_metrics']['pressures']
            )
            del metrics['compression_metrics']['pressures']  # Don't save full list
        
        if metrics['compression_metrics']['hold_times']:
            metrics['compression_metrics']['avg_hold_time'] = np.mean(
                metrics['compression_metrics']['hold_times']
            )
            del metrics['compression_metrics']['hold_times']  # Don't save full list
        
        try:
            with open(filepath, 'w') as f:
                json.dump(metrics, f, indent=2)
        except Exception as e:
            print(f"Error saving daily metrics: {e}")
    
    def _update_stand_metrics(self, is_confirmed: bool, sbp_drop: float) -> None:
        """Update daily stand metrics"""
        self.daily_metrics['stand_events']['total_attempts'] += 1
        
        if is_confirmed:
            self.daily_metrics['stand_events']['confirmed'] += 1
        else:
            self.daily_metrics['stand_events']['false_positive'] += 1
    
    def _update_bp_metrics(self, sbp_drop: float) -> None:
        """Update daily BP metrics"""
        self.daily_metrics['bp_metrics']['drops_recorded'] += 1
        self.daily_metrics['bp_metrics']['drops_list'].append(sbp_drop)
    
    def _update_compression_metrics(self, pressure: float, hold_time: float, dose: float) -> None:
        """Update daily compression metrics"""
        self.daily_metrics['compression_metrics']['cycles'] += 1
        self.daily_metrics['compression_metrics']['total_dose'] += dose
        self.daily_metrics['compression_metrics']['pressures'].append(pressure)
        self.daily_metrics['compression_metrics']['hold_times'].append(hold_time)
    
    def _update_feedback_metrics(self, rating: Optional[int], comment: str) -> None:
        """Update daily feedback metrics"""
        self.daily_metrics['user_feedback']['count'] += 1
        
        if rating is not None:
            self.daily_metrics['user_feedback']['ratings'].append(rating)
        
        if comment:
            self.daily_metrics['user_feedback']['comments'].append(comment)
    
    def _update_alert_metrics(self, alert_type: str) -> None:
        """Update daily alert metrics"""
        self.daily_metrics['system_alerts']['count'] += 1
        
        if alert_type not in self.daily_metrics['system_alerts']['types']:
            self.daily_metrics['system_alerts']['types'][alert_type] = 0
        
        self.daily_metrics['system_alerts']['types'][alert_type] += 1
    
    def get_recent_events(self, n: int = 10, event_type: Optional[EventType] = None) -> List[Dict]:
        """
        Get recent events from buffer
        
        Args:
            n: Number of events to return
            event_type: Filter by event type (optional)
            
        Returns:
            List of event dictionaries
        """
        events = list(self.event_buffer)
        
        # Filter by type if specified
        if event_type:
            events = [e for e in events if e['event_type'] == event_type.value]
        
        # Return most recent n events
        return events[-n:]
    
    def get_daily_metrics(self) -> Dict:
        """
        Get current daily metrics
        
        Returns:
            Dictionary of daily aggregated metrics
        """
        metrics = self.daily_metrics.copy()
        
        # Calculate real-time averages
        if metrics['bp_metrics']['drops_list']:
            metrics['bp_metrics']['avg_drop'] = float(np.mean(metrics['bp_metrics']['drops_list']))
            metrics['bp_metrics']['max_drop'] = float(np.max(metrics['bp_metrics']['drops_list']))
            metrics['bp_metrics']['min_drop'] = float(np.min(metrics['bp_metrics']['drops_list']))
        
        if metrics['compression_metrics']['pressures']:
            metrics['compression_metrics']['avg_pressure'] = float(np.mean(
                metrics['compression_metrics']['pressures']
            ))
        
        if metrics['compression_metrics']['hold_times']:
            metrics['compression_metrics']['avg_hold_time'] = float(np.mean(
                metrics['compression_metrics']['hold_times']
            ))
        
        if metrics['user_feedback']['ratings']:
            metrics['user_feedback']['avg_rating'] = float(np.mean(
                metrics['user_feedback']['ratings']
            ))
        
        return metrics
    
    def export_events(self, start_date: Optional[date] = None, 
                     end_date: Optional[date] = None) -> List[Dict]:
        """
        Export events within date range
        
        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            
        Returns:
            List of event dictionaries
        """
        # Implementation would read from disk
        # For now, return buffer
        return list(self.event_buffer)
    
    def clear_buffer(self) -> None:
        """Clear in-memory event buffer"""
        self.event_buffer.clear()

    def load_events_from_disk(self, max_events: Optional[int] = None) -> int:
        """
        Load events from JSON files in log_directory into the event buffer.
        Loads the MOST RECENT events (by timestamp) to ensure current data is available.

        Args:
            max_events: Maximum number of events to load (None for no limit, up to buffer_size)

        Returns:
            Number of events loaded
        """
        import glob

        # Filter out timestamps before year 2000 (to exclude 1969 bugs and invalid data)
        # Unix timestamp for Jan 1, 2000 00:00:00 UTC is 946684800
        MIN_VALID_TIMESTAMP = 946684800

        # Get all JSON event files (exclude daily_metrics files)
        json_files = glob.glob(os.path.join(self.log_directory, "*.json"))
        event_files = [f for f in json_files if not os.path.basename(f).startswith("daily_metrics_")]

        # Load all events into a temporary list and sort by timestamp (newest first)
        temp_events = []
        for filepath in event_files:
            try:
                with open(filepath, 'r') as f:
                    event = json.load(f)

                # Only load stand events with valid timestamps
                if event.get('event_type') in ['stand_confirmed', 'stand_false_positive']:
                    timestamp = event.get('timestamp', 0)
                    if timestamp >= MIN_VALID_TIMESTAMP:
                        temp_events.append(event)

                        # Update event counter to avoid ID collisions
                        if 'event_id' in event:
                            self.event_counter = max(self.event_counter, event['event_id'])

            except (json.JSONDecodeError, KeyError) as e:
                # Skip corrupted files
                continue

        # Sort by timestamp (newest first)
        temp_events.sort(key=lambda e: e.get('timestamp', 0), reverse=True)

        # Take the most recent events up to max_events
        if max_events:
            temp_events = temp_events[:max_events]

        # Put them back in chronological order for the buffer
        temp_events.sort(key=lambda e: e.get('timestamp', 0))

        # Clear buffer and add sorted events
        self.event_buffer = []
        for event in temp_events:
            self.event_buffer.append(event)

        return len(self.event_buffer)


if __name__ == "__main__":
    # Test the event logger
    import time
    
    print("Testing EventLogger...")
    
    logger = EventLogger(log_directory="./test_logs", buffer_size=50)
    
    # Test stand event
    print("\n1. Logging confirmed stand event...")
    event1 = logger.log_stand_event(
        is_confirmed=True,
        timestamp=time.time(),
        pitch_angle=72.5,
        sbp_drop=18.2,
        posture_state="Standing",
        bp_baseline=125.0,
        bp_current=106.8,
        recovery_time=8.5,
        max_drop=22.1
    )
    print(f"   Event ID: {event1['event_id']}")
    print(f"   Type: {event1['event_type']}")
    
    # Test false positive
    print("\n2. Logging false-positive stand...")
    event2 = logger.log_stand_event(
        is_confirmed=False,
        timestamp=time.time(),
        pitch_angle=45.0,
        sbp_drop=5.0,
        posture_state="Seated",
        bp_baseline=120.0,
        bp_current=115.0
    )
    print(f"   Event ID: {event2['event_id']}")
    print(f"   False positive: {not event2['is_confirmed']}")
    
    # Test compression cycle
    print("\n3. Logging compression cycle...")
    event3 = logger.log_compression_cycle(
        timestamp=time.time(),
        target_pressure=20.0,
        achieved_pressure=19.5,
        hold_duration=60.0,
        release_rate="Medium",
        cycle_dose=1170.0,
        was_auto=True
    )
    print(f"   Event ID: {event3['event_id']}")
    print(f"   Dose: {event3['compression_params']['cycle_dose']} kPa·s")
    
    # Test user feedback
    print("\n4. Logging user feedback...")
    event4 = logger.log_user_feedback(
        timestamp=time.time(),
        feedback_type="comfort",
        rating=4,
        comment="Compression was comfortable",
        related_event_id=event3['event_id']
    )
    print(f"   Event ID: {event4['event_id']}")
    print(f"   Rating: {event4['feedback']['rating']}/5")
    
    # Get daily metrics
    print("\n5. Daily metrics:")
    metrics = logger.get_daily_metrics()
    print(f"   Stand events: {metrics['stand_events']}")
    print(f"   BP drops: {metrics['bp_metrics']['drops_recorded']}")
    print(f"   Compression cycles: {metrics['compression_metrics']['cycles']}")
    print(f"   User feedback: {metrics['user_feedback']['count']}")
    
    # Get recent events
    print("\n6. Recent events:")
    recent = logger.get_recent_events(n=5)
    for event in recent:
        print(f"   [{event['event_id']}] {event['event_type']} at "
              f"{datetime.fromtimestamp(event['timestamp']).strftime('%H:%M:%S')}")
    
    print("\n✓ EventLogger working correctly")
    print(f"\nEvents saved to: {logger.log_directory}")
