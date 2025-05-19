import numpy as np
from problems.composite_prob import CompositeProblem

class Ridge(CompositeProblem):
    
    def __init__(self, lbd):
        self.lbd = lbd
        
    def g_gradient(self, X, y, w):
        # X.T (X*w - y) + lbd*w
        return X.T @ (X @ w - y) + self.lbd*w
    
    def h_proximal_op(self, w, t):
        # h part smooth so no proximal step 
        return w

    def obj_value(self, X, y, w):
        # 1/2*||Xw-y||_2^2 + ldb/2*||w||^2_2
        return 0.5 * np.linalg.norm(X @ w - y)**2 + (0.5*self.lbd) * np.linalg.norm(w, ord=2)**2