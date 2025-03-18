from abc import ABC, abstractmethod
import numpy as np

class Solver(ABC):
    """
    Base solver abstract class
    
    params:
        - max_iter
        - eps
    """
    
    def __init__(self, max_iter=1000, eps=1e-5):
        self.max_iter = max_iter
        self.eps = eps
        self.loss_history = []
    
    @abstractmethod
    def solve(self, A, b):
        pass
    
    def _check_convergence(self, x_old, x_new):
        return np.linalg.norm(x_new - x_old) < self.eps
    
    def get_loss_history(self):
        return self.loss_history