import numpy as np
from problems.composite_prob import CompositeProblem

class Lasso(CompositeProblem):
    
    def __init__(self, A, b, lbd):
        self.A = A  # features
        self.b = b  # target
        self.lbd = lbd
        
    def gradient(self, x):
        # A.T (A*x - b)
        return self.A.T @ (self.A@x - self.b)
    
    def proximal_op(self, x, t):
        # shrinkage operator
        return np.sign(x) * np.maximum(np.abs(x) - t*self.lbd, 0)

    def obj_value(self, x):
        pass