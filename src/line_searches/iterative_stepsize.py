import numpy as np

from .line_search import LineSearch
    
class IterativeStepSize(LineSearch):
    
    def __init__(self, X):
        self.L = np.linalg.norm(X, ord=2)**2
        
    def compute_stepsize(self) -> int:
        #TODO: compute 1/L in init and return constant value instead of doing division evey algo iteration
        return 1/self.L 