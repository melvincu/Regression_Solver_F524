from abc import ABC, abstractmethod
import numpy as np

class BaseSolver(ABC):
    """
    Base solver abstract class
    
    params:
        - max_iter
        - eps
    """
    
    def __init__(self, max_iter=1000, eps=1e-6):
        self.max_iter = max_iter
        self.eps = eps
        self.loss_history = []
    
    @abstractmethod
    def solve(self, X, y):
        pass
    
    def _check_convergence(self, x_old, x_new):
        return np.linalg.norm(x_new - x_old) < self.eps