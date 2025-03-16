import numpy as np
from ..line_search import *
from ..proximal_operators import BaseProxOp
from base_solver import BaseSolver

class ISTA(BaseSolver):
    def __init__(self, line_search:LineSearch , prox_op:BaseProxOp, **kwargs):
        super().__init__(**kwargs)
        self.line_search = line_search
        self.prox_op = prox_op
    
    def solve(self, X, y):
        # L = np.linalg.norm(self.X, ord=2)**2 # constant step size !
        
        N = X.shape[1] # num features
        x = np.zeros(N) # x0
        for k in range(self.max_iter):
            grad = X.T @ (X @ x - y) # check grad computation
            t = 1 # TODO: line search
            x_new = self.prox_op.apply(x-t*grad)
            if self._check_convergence(x, x_new): break
            x = x_new
            
            print(f"iter {k}: fval={0} / norm_grad={0}")
        return x