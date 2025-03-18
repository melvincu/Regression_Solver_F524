import numpy as np
from line_searches.line_search import *
from .solver import Solver

class GradientDesc(Solver):
    def __init__(self, line_search:LineSearch, **kwargs):
        super().__init__(**kwargs)
        self.line_search = line_search
    
    def solve(self, X, y):        
        n = X.shape[1] # num features
        x = np.zeros((n,1)) # w0 (nx1)

        for _ in range(self.max_iter):
            
            # compute step size t_k
            t_k = 0.01
            
            # gradient descent step
            grad = X.T @ (X@x - y)            
            x_new = x-t_k*grad
            
            # compute loss
            loss = 0.5 * np.linalg.norm(X @ x_new - y)**2 # TODO: maybe do a compute loss method in baseSolver abstract class
            self.loss_history.append(loss)
            
            # stop criterion
            if self._check_convergence(x, x_new): break            
            x = x_new
            
        return x
