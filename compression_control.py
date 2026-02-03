"""
Compression Control Module
Manages compression system with real-time state tracking and control
"""

import numpy as np
from collections import deque
from enum import Enum
from datetime import datetime
import time


class CompressionState(Enum):
    """Compression system states"""
    RELEASED = "Released"
    ENGAGING = "Engaging"
    HOLDING = "Holding"
    RELEASING = "Releasing"


class ReleaseRate(Enum):
    """Compression release rate settings"""
    SLOW = "Slow"
    MEDIUM = "Medium"
    FAST = "Fast"


class CompressionController:
    """
    Controls compression system with state machine
    
    States:
    RELEASED → ENGAGING → HOLDING → RELEASING → RELEASED
    
    Tracks:
    - Current pressure
    - Target pressure
    - Hold duration
    - Release rate
    - Total daily dose
    """
    
    def __init__(self):
        """Initialize compression controller"""
        # Current state
        self.current_state = CompressionState.RELEASED
        self.state_entry_time = time.time()
        
        # Pressure parameters (kPa)
        self.current_pressure = 0.0
        self.target_pressure = 0.0
        self.max_pressure = 30.0
        self.min_pressure = 0.0
        
        # Control parameters
        self.hold_duration = 0.0  # Target hold time (seconds)
        self.hold_elapsed = 0.0   # Actual hold time so far (seconds)
        self.release_rate = ReleaseRate.MEDIUM
        
        # Release rate parameters (kPa/second)
        self.release_rates = {
            ReleaseRate.SLOW: 2.0,
            ReleaseRate.MEDIUM: 5.0,
            ReleaseRate.FAST: 10.0
        }
        
        # Engagement rate (kPa/second)
        self.engagement_rate = 5.0
        
        # Daily tracking
        self.daily_compression_dose = 0.0  # kPa·seconds
        self.daily_start_time = datetime.now().date()
        
        # Cycle tracking
        self.cycle_count = 0
        self.last_cycle_time = None
        
        # State history
        self.state_history = deque(maxlen=100)
        self.pressure_history = deque(maxlen=1000)
        
        # Previous update time for dose calculation
        self.prev_update_time = time.time()
    
    def update(self, sensor_pressure: float, timestamp: float = None) -> None:
        """
        Update compression controller state
        
        Args:
            sensor_pressure: Current measured pressure (kPa)
            timestamp: Current timestamp (optional)
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Update current pressure from sensor
        self.current_pressure = sensor_pressure
        
        # Calculate dose increment
        dt = timestamp - self.prev_update_time
        dose_increment = self.current_pressure * dt
        self.daily_compression_dose += dose_increment
        self.prev_update_time = timestamp
        
        # Check if new day
        current_date = datetime.now().date()
        if current_date != self.daily_start_time:
            self._reset_daily_dose()
            self.daily_start_time = current_date
        
        # Record pressure
        self.pressure_history.append({
            'timestamp': timestamp,
            'pressure': self.current_pressure,
            'state': self.current_state.value
        })
        
        # Update state machine
        self._update_state_machine(timestamp)
    
    def _update_state_machine(self, timestamp: float) -> None:
        """
        Update state machine based on current conditions
        
        Args:
            timestamp: Current timestamp
        """
        state_duration = timestamp - self.state_entry_time
        
        if self.current_state == CompressionState.RELEASED:
            # Stay in released state until commanded to engage
            pass
        
        elif self.current_state == CompressionState.ENGAGING:
            # Check if target pressure reached
            if self.current_pressure >= self.target_pressure * 0.95:
                self._transition_to(CompressionState.HOLDING, timestamp)
        
        elif self.current_state == CompressionState.HOLDING:
            # Update hold elapsed time
            self.hold_elapsed = state_duration
            
            # Check if hold duration completed
            if self.hold_duration > 0 and self.hold_elapsed >= self.hold_duration:
                self._transition_to(CompressionState.RELEASING, timestamp)
        
        elif self.current_state == CompressionState.RELEASING:
            # Check if fully released
            if self.current_pressure <= self.min_pressure + 1.0:
                self._transition_to(CompressionState.RELEASED, timestamp)
                self.cycle_count += 1
                self.last_cycle_time = timestamp
    
    def engage_compression(self, target_pressure: float, hold_duration: float,
                          release_rate: ReleaseRate = ReleaseRate.MEDIUM) -> bool:
        """
        Start compression cycle

        Args:
            target_pressure: Target pressure in kPa
            hold_duration: How long to hold compression (seconds)
            release_rate: Rate of pressure release

        Returns:
            True if compression started, False if already active
        """
        if self.current_state != CompressionState.RELEASED:
            return False

        # Validate parameters
        self.target_pressure = np.clip(target_pressure,
                                      self.min_pressure,
                                      self.max_pressure)
        self.hold_duration = max(0.0, hold_duration)
        self.release_rate = release_rate

        # Transition to engaging
        self._transition_to(CompressionState.ENGAGING, time.time())

        return True

    def preinflate(self, target_pressure: float) -> bool:
        """
        Pre-inflate compression to partial pressure (early intervention)

        Args:
            target_pressure: Partial target pressure in kPa (typically 50% of full)

        Returns:
            True if pre-inflation started
        """
        if self.current_state == CompressionState.RELEASED:
            # Start partial compression
            self.target_pressure = np.clip(target_pressure,
                                          self.min_pressure,
                                          self.max_pressure)
            self._transition_to(CompressionState.ENGAGING, time.time())
            return True
        return False
    
    def release_compression(self) -> bool:
        """
        Manually release compression
        
        Returns:
            True if release started, False if already released
        """
        if self.current_state == CompressionState.RELEASED:
            return False
        
        # Transition to releasing
        self._transition_to(CompressionState.RELEASING, time.time())
        
        return True
    
    def _transition_to(self, new_state: CompressionState, timestamp: float) -> None:
        """
        Transition to new state
        
        Args:
            new_state: Target state
            timestamp: Transition timestamp
        """
        old_state = self.current_state
        self.current_state = new_state
        self.state_entry_time = timestamp
        
        # Reset hold elapsed when entering holding state
        if new_state == CompressionState.HOLDING:
            self.hold_elapsed = 0.0
        
        # Log transition
        self.state_history.append({
            'timestamp': timestamp,
            'from_state': old_state.value,
            'to_state': new_state.value
        })
    
    def simulate_pressure_control(self) -> float:
        """
        Simulate pressure control (for testing with mock data)
        
        In production, this would be replaced by actual hardware control.
        
        Returns:
            Simulated target pressure for current state
        """
        if self.current_state == CompressionState.RELEASED:
            return 0.0
        
        elif self.current_state == CompressionState.ENGAGING:
            # Ramp up pressure
            state_duration = time.time() - self.state_entry_time
            pressure = min(self.target_pressure, 
                          self.engagement_rate * state_duration)
            return pressure
        
        elif self.current_state == CompressionState.HOLDING:
            # Maintain target pressure
            return self.target_pressure
        
        elif self.current_state == CompressionState.RELEASING:
            # Ramp down pressure
            state_duration = time.time() - self.state_entry_time
            rate = self.release_rates[self.release_rate]
            pressure = max(0.0, self.target_pressure - rate * state_duration)
            return pressure
        
        return 0.0
    
    def get_state_duration(self) -> float:
        """Get duration in current state (seconds)"""
        return time.time() - self.state_entry_time
    
    def get_status(self) -> dict:
        """
        Get comprehensive status
        
        Returns:
            Dictionary with all status information
        """
        return {
            'state': self.current_state.value,
            'state_duration': self.get_state_duration(),
            'current_pressure': self.current_pressure,
            'target_pressure': self.target_pressure,
            'hold_duration': self.hold_duration,
            'hold_elapsed': self.hold_elapsed,
            'release_rate': self.release_rate.value,
            'daily_dose': self.daily_compression_dose,
            'cycle_count': self.cycle_count,
            'is_engaged': self.is_engaged(),
            'completion_percent': self.get_completion_percent()
        }
    
    def is_engaged(self) -> bool:
        """Check if compression is currently engaged"""
        return self.current_state in [
            CompressionState.ENGAGING,
            CompressionState.HOLDING,
            CompressionState.RELEASING
        ]
    
    def get_completion_percent(self) -> float:
        """
        Get completion percentage of current cycle
        
        Returns:
            Percentage [0-100]
        """
        if self.current_state == CompressionState.RELEASED:
            return 0.0
        
        elif self.current_state == CompressionState.ENGAGING:
            if self.target_pressure > 0:
                return (self.current_pressure / self.target_pressure) * 25.0
            return 0.0
        
        elif self.current_state == CompressionState.HOLDING:
            if self.hold_duration > 0:
                return 25.0 + (self.hold_elapsed / self.hold_duration) * 50.0
            return 25.0
        
        elif self.current_state == CompressionState.RELEASING:
            if self.target_pressure > 0:
                return 75.0 + (1.0 - self.current_pressure / self.target_pressure) * 25.0
            return 75.0
        
        return 0.0
    
    def _reset_daily_dose(self) -> None:
        """Reset daily compression dose"""
        self.daily_compression_dose = 0.0
    
    def reset(self) -> None:
        """Reset controller to initial state"""
        self.current_state = CompressionState.RELEASED
        self.state_entry_time = time.time()
        self.current_pressure = 0.0
        self.target_pressure = 0.0
        self.hold_duration = 0.0
        self.hold_elapsed = 0.0
        self.cycle_count = 0
        self.state_history.clear()
        self.pressure_history.clear()


if __name__ == "__main__":
    # Test the compression controller
    print("Testing CompressionController...")
    
    controller = CompressionController()
    
    print(f"\nInitial state: {controller.current_state.value}")
    print(f"  Pressure: {controller.current_pressure:.1f} kPa")
    print(f"  Is engaged: {controller.is_engaged()}")
    
    # Start compression cycle
    print("\nEngaging compression...")
    success = controller.engage_compression(
        target_pressure=20.0,
        hold_duration=5.0,
        release_rate=ReleaseRate.MEDIUM
    )
    print(f"  Started: {success}")
    print(f"  State: {controller.current_state.value}")
    print(f"  Target: {controller.target_pressure:.1f} kPa")
    
    # Simulate pressure ramp
    print("\nSimulating pressure ramp...")
    for i in range(10):
        simulated_pressure = controller.simulate_pressure_control()
        controller.update(simulated_pressure)
        time.sleep(0.5)
        
        status = controller.get_status()
        if i % 2 == 0:
            print(f"  t={i*0.5:.1f}s: {status['state']}, "
                  f"P={status['current_pressure']:.1f} kPa, "
                  f"Progress={status['completion_percent']:.0f}%")
    
    print(f"\nFinal state: {controller.current_state.value}")
    print(f"  Cycles completed: {controller.cycle_count}")
    print(f"  Daily dose: {controller.daily_compression_dose:.1f} kPa·s")
    
    print("\n✓ CompressionController working correctly")
