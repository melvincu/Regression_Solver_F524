import numpy as np
from problems.composite_prob import CompositeProblem

class ElasticNet(CompositeProblem):
    
    def __init__(self, X, y, lbd1, lbd2):
        self.X = X
        self.y = y
        self.lbd1 = lbd1
        self.lbd2 = lbd2
        
    def g_gradient(self, w):
        # X.T (X*w - y) + lb2*w
        return self.X.T @ (self.X@w - self.y) + self.lbd2*w
    
    def h_proximal_op(self, w, t):
        # shrinkage operator
        return np.sign(w) * np.maximum(np.abs(w) - t*self.lbd1, 0)

    def obj_value(self, w):
        # 1/2*||Xw-y||_2^2 + 1/2*lbd2||w||_2^2 + lbd1||w||_1
        return 0.5 * np.linalg.norm(self.X @ w - self.y)**2 + \
            (0.5 * self.lbd2 * np.linalg.norm(w)**2) + \
            (self.lbd1 * np.linalg.norm(w, ord=1))