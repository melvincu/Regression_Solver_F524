from abc import ABC, abstractmethod

class DataHandler(ABC):
    """Data handler class"""

    @abstractmethod
    def get_data(self):
        pass