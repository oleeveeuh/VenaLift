"""
Main monitoring orchestrator.

Coordinates all system components:
- Data collection from sensors
- Feature extraction
- Posture and stand detection
- ML classification
- Compression control
- Logging

Provides a simple interface for real-time monitoring.
"""

import time
from pathlib import Path
from typing import Optional, Callable, Union

from .data.source import DataSource, DataBuffer, MockDataSource
from .data.models import SensorReading, StandEvent
from .detection.detector import PostureDetector, StandDetector, PostureEstimate
from .ml.classifier import StandClassifier
from .control.controller import CompressionController, MockCompressionDevice, CompressionPolicy, CompressionCommand
from .logging.logger import CombinedLogger


class MonitoringSystem:
    """
    Main physiological monitoring system.

    Orchestrates all components in a real-time processing loop.
    """

    def __init__(
        self,
        data_source: DataSource,
        sample_rate: float = 50.0,
        log_dir: Optional[Union[Path, str]] = None,
        compression_policy: Optional[CompressionPolicy] = None,
    ):
        """
        Initialize monitoring system.

        Args:
            data_source: Data source (mock or live)
            sample_rate: Sampling rate in Hz
            log_dir: Directory for log files (None to disable logging)
            compression_policy: Compression control policy (None for default)
        """
        self.sample_rate = sample_rate
        self.running = False

        # Data layer
        self.data_source = data_source
        self.data_buffer = DataBuffer(duration=5.0, sample_rate=sample_rate)

        # Detection layer
        self.posture_detector = PostureDetector(sample_rate)
        self.stand_detector = StandDetector(sample_rate)

        # ML layer
        self.classifier = StandClassifier()

        # Control layer
        self.compression_controller = CompressionController(compression_policy)
        self.compression_device = MockCompressionDevice()  # Replace with real device

        # Logging layer
        self.logger = CombinedLogger(log_dir) if log_dir else None

        # Callbacks for external integration
        self.on_stand_detected: Optional[Callable[[StandEvent], None]] = None
        self.on_compression_triggered: Optional[Callable[[CompressionCommand], None]] = None
        self.on_posture_change: Optional[Callable[[PostureEstimate], None]] = None

        # State tracking
        self._last_posture = None
        self._sample_count = 0
        self._start_time = None

    def start(self) -> None:
        """Start the monitoring system."""
        self.running = True
        self._start_time = time.time()
        print(f"Monitoring system started at {self.sample_rate} Hz")

    def stop(self) -> None:
        """Stop the monitoring system and cleanup."""
        self.running = False

        # Finalize logs
        if self.logger:
            self.logger.close()

        # Close data source
        self.data_source.close()

        print("Monitoring system stopped")

    def process_one(self) -> dict:
        """
        Process a single sensor reading.

        Returns:
            Summary of processing results
        """
        # Read sensor data
        reading = self.data_source.read()
        self.data_buffer.add(reading)

        # Get recent window for posture detection
        recent = self.data_buffer.get_latest(int(self.sample_rate))

        # Detect posture
        posture = self.posture_detector.update(recent)

        # Check for posture change
        if self._last_posture != posture.state:
            if self.on_posture_change:
                self.on_posture_change(posture)
            self._last_posture = posture.state

        # Detect stand events
        event = self.stand_detector.update(reading)

        # Process event if detected
        result = {
            "timestamp": reading.timestamp,
            "posture": posture.state.value,
            "posture_confidence": posture.confidence,
            "event_detected": event is not None,
        }

        if event:
            # Classify event
            is_valid, confidence, explanation = self.classifier.classify(event)

            # Update event with classification
            event.is_valid = is_valid
            event.confidence = confidence

            # Log event
            compression_triggered = False
            if self.logger:
                bp_sys = reading.blood_pressure.systolic if reading.blood_pressure else None
                bp_dia = reading.blood_pressure.diastolic if reading.blood_pressure else None
                hr = reading.blood_pressure.heart_rate if reading.blood_pressure else None

                self.logger.log(
                    event,
                    compression_triggered=False,  # Will update below
                    bp_systolic=bp_sys,
                    bp_diastolic=bp_dia,
                    heart_rate=hr,
                )

            # Trigger compression if valid
            command = None
            if is_valid:
                command = self.compression_controller.process_event(event, reading.timestamp)
                if command:
                    compression_triggered = True
                    self.compression_device.execute(command, reading.timestamp)

                    # Update log with compression
                    if self.logger:
                        # Re-log with compression flag
                        self.logger.log(
                            event,
                            compression_triggered=True,
                            bp_systolic=bp_sys,
                            bp_diastolic=bp_dia,
                            heart_rate=hr,
                        )

            # Update compression device state
            device_command = self.compression_controller.update_device_state(reading.timestamp)
            if device_command:
                self.compression_device.execute(device_command, reading.timestamp)

            # Trigger callbacks
            if self.on_stand_detected:
                self.on_stand_detected(event)

            if command and self.on_compression_triggered:
                self.on_compression_triggered(command)

            # Add to result
            result.update({
                "event_valid": is_valid,
                "event_confidence": confidence,
                "compression_triggered": compression_triggered,
                "explanation": explanation,
            })

        self._sample_count += 1

        return result

    def run_until(
        self,
        duration: Optional[float] = None,
        max_samples: Optional[int] = None,
        verbose: bool = True,
    ) -> None:
        """
        Run monitoring loop for a specified duration or sample count.

        Args:
            duration: Maximum run time in seconds (None for infinite)
            max_samples: Maximum samples to process (None for infinite)
            verbose: Print status updates
        """
        self.start()

        start_time = time.time()
        samples = 0

        try:
            while self.running:
                # Check stopping conditions
                if duration and (time.time() - start_time) >= duration:
                    break
                if max_samples and samples >= max_samples:
                    break

                # Process one sample
                result = self.process_one()
                samples += 1

                # Print updates
                if verbose and samples % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = samples / elapsed
                    print(f"Processed {samples} samples ({rate:.1f} Hz)")
                    print(f"  Posture: {result['posture']} ({result['posture_confidence']:.2f})")

                    # Get daily summary
                    if self.logger:
                        summary = self.logger.get_daily_summary()
                        if summary:
                            print(f"  Today: {summary.valid_stands} valid stands, "
                                  f"{summary.false_positives} false positives")

                # Maintain sample rate timing
                time.sleep(1.0 / self.sample_rate)

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        finally:
            self.stop()

    def get_status(self) -> dict:
        """Get current system status."""
        elapsed = time.time() - self._start_time if self._start_time else 0

        status = {
            "running": self.running,
            "sample_rate": self.sample_rate,
            "samples_processed": self._sample_count,
            "elapsed_time": elapsed,
            "actual_rate": self._sample_count / elapsed if elapsed > 0 else 0,
            "current_posture": self._last_posture.value if self._last_posture else "unknown",
        }

        # Add compression status
        status["compression"] = self.compression_controller.get_status()

        # Add daily summary if logging
        if self.logger:
            summary = self.logger.get_daily_summary()
            if summary:
                status["daily_summary"] = {
                    "valid_stands": summary.valid_stands,
                    "false_positives": summary.false_positives,
                    "avg_duration": summary.avg_stand_duration,
                }

        return status


def create_mock_system(
    sample_rate: float = 50.0,
    log_dir: Optional[Union[Path, str]] = None,
    event_interval: float = 30.0,
) -> MonitoringSystem:
    """
    Create a monitoring system with mock data source.

    Convenience function for testing and development.

    Args:
        sample_rate: Sampling rate in Hz
        log_dir: Directory for log files
        event_interval: Average seconds between mock stand events

    Returns:
        Configured MonitoringSystem instance
    """
    from .data.source import MockDataSource

    data_source = MockDataSource(
        sample_rate=sample_rate,
        event_interval=event_interval,
    )

    return MonitoringSystem(
        data_source=data_source,
        sample_rate=sample_rate,
        log_dir=log_dir,
    )


def create_live_system(
    sensor_config: dict,
    sample_rate: float = 50.0,
    log_dir: Optional[Union[Path, str]] = None,
) -> MonitoringSystem:
    """
    Create a monitoring system with live sensor connections.

    Args:
        sensor_config: Configuration dict for live sensors
        sample_rate: Sampling rate in Hz
        log_dir: Directory for log files

    Returns:
        Configured MonitoringSystem instance

    Note:
        This is a template - implement actual sensor connections
        by creating a LiveDataSource subclass.
    """
    # TODO: Implement live sensor connections
    raise NotImplementedError(
        "Live sensor integration requires implementing LiveDataSource. "
        "See src/data/source.py for the template."
    )
