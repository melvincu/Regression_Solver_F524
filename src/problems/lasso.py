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
        # 1/2*||Ax-b||_2^2 + lbd||x||_1
        return 0.5 * np.linalg.norm(self.A @ x - self.b)**2 + self.lbd * np.linalg.norm(x, ord=1)