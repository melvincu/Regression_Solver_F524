from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

class LineSearch(ABC):
    """Abstract base class for line search methods."""
    
    @abstractmethod
    def compute_stepsize(self) -> int:
        """Compute step size baes on underlying strategy (child class)"""
        pass

