import numpy as np

from .line_search import LineSearch

class FixedStepSize(LineSearch):
    
    def __init__(self, A):
        self.t = 1/(np.linalg.norm(A, ord=2)**2) # 1/L (L=||A||_2^2)
        
    def compute_stepsize(self) -> int:
        """t_k = 1/L (fixed)"""
        return self.t