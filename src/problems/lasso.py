import numpy as np
from problems.composite_prob import CompositeProblem

class Lasso(CompositeProblem):
    
    def __init__(self, lbd):
        self.lbd = lbd
    
    def g_gradient(self, X, y, w):
        # X.T (X*w - y)
        return X.T @ (X @ w - y)
    
    def h_proximal_op(self, w, t):
        # shrinkage operator
        return np.sign(w) * np.maximum(np.abs(w) - t*self.lbd, 0)

    def obj_value(self, X, y, w):
        # 1/2*||Xw-y||_2^2 + lbd||w||_1
        return 0.5 * np.linalg.norm(X @ w - y)**2 + self.lbd * np.linalg.norm(w, ord=1)