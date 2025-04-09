import numpy as np
from problems.composite_prob import CompositeProblem

class ElasticNet(CompositeProblem):
    
    def __init__(self, A, b, lbd1, lbd2):
        self.A = A
        self.b = b
        self.lbd1 = lbd1
        self.lbd2 = lbd2
        
    def g_gradient(self, x):
        # A.T (A*x - b) + lb2*x
        return self.A.T @ (self.A@x - self.b) + self.lbd2*x
    
    def h_proximal_op(self, x, t):
        # shrinkage operator
        return np.sign(x) * np.maximum(np.abs(x) - t*self.lbd1, 0)

    def obj_value(self, x):
        # 1/2*||Ax-b||_2^2 + 1/2*lbd2||x||_2^2 + lbd1||x||_1
        return 0.5 * np.linalg.norm(self.A @ x - self.b)**2 + \
            (0.5 * self.lbd2 * np.linalg.norm(x)**2) + \
            (self.lbd1 * np.linalg.norm(x, ord=1))