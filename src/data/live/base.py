# data_sources/base.py
from abc import ABC, abstractmethod

class SensorDataSource(ABC):
    """
    Abstract base class for all sensor data sources.
    """

    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def disconnect(self):
        pass

    @abstractmethod
    def read(self):
        """
        Return a dict of sensor values.
        """
        pass
