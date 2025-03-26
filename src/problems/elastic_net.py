import numpy as np
from problems.composite_prob import CompositeProblem

class ElasticNet(CompositeProblem):
    
    def __init__(self, A, b, lbd1, lbd2):
        self.A = A
        self.b = b
        self.lbd1 = lbd1
        self.lbd2 = lbd2
        
    def gradient(self, x):
        # A.T (A*x - b) + lb2*x
        return self.A.T @ (self.A@x - self.b) + self.lbd2*x
    
    def proximal_op(self, x, t):
        # shrinkage operator
        return np.sign(x) * np.maximum(np.abs(x) - t*self.lbd1, 0)
    
    def obj_value(self, x):
        pass