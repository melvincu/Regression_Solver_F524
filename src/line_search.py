from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

class LineSearch(ABC):
    """Abstract base class for line search methods."""
    
    @abstractmethod
    def compute_stepsize(self) -> int:
        """Compute step size based on current iterate and gradient."""
        pass

##############################################
# Concrete impl.
##############################################

class BacktrackingLS(LineSearch):
    """Backtracking line search method (Armijo rule)."""
    
    def __init__(self, alpha, c):
        self.alpha = alpha  # Armijo parameter (sufficient decrease)
        self.c = c    # Reduction factor
    
    def compute_stepsize(self, X, x) -> int:
        # t = 1.0  # Start with a large step
        # while np.linalg.norm(X.T @ (X @ (w - t * grad) - y)) > self.alpha * np.linalg.norm(grad):
        #     t *= self.beta  # Reduce step size
        # return t
        pass

class FixedStepSize(LineSearch):
    
    def __init__(self, X):
        self.L = np.linalg.norm(X, ord=2)**2
        
    def compute_stepsize(self) -> int:
        #TODO: compute 1/L in init and return constant value instead of doing division evey algo iteration
        return 1/self.L 