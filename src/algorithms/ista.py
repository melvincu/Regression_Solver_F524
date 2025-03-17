import numpy as np
from line_search import *
from regulirizers import Regularizer
from .solver import Solver

class ISTA(Solver):
    def __init__(self, line_search:LineSearch , regulirizer:Regularizer, **kwargs):
        super().__init__(**kwargs)
        self.line_search = line_search
        self.regulirizer = regulirizer
    
    def solve(self, X, y):        
        n = X.shape[1] # num features
        x = np.zeros((n,1)) # w0 (nx1)

        for _ in range(self.max_iter):
            
            # compute step size t_k
            t_k = 0.01
            
            # gradient descent step
            grad = X.T @ (X@x - y) # TODO: epxlain in report computations and formulations    
            x_new = self.regulirizer.prox_op(x-t_k*grad, t_k)

            # compute loss (total loss = data loss + reg loss)
            loss = 0.5 * np.linalg.norm(X @ x_new - y)**2 + self.regulirizer.compute_reg_loss(x)            
            self.loss_history.append(loss)
            
            # stop criterion
            if self._check_convergence(x, x_new): break
            x = x_new

        return x
