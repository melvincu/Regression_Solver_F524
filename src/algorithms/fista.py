import numpy as np
from line_searches.line_search import *
from regulirizers import Regularizer
from .solver import Solver

class FISTA(Solver):
    def __init__(self, line_search:LineSearch , regulirizer:Regularizer, **kwargs):
        super().__init__(**kwargs)
        self.line_search = line_search
        self.regulirizer = regulirizer
    
    def solve(self, A, b):      
        n = A.shape[1] # num features
        x = np.zeros((n,1)) # w0 (nx1)
        y_k = np.zeros((n,1)) 
        t_k = self.line_search.compute_stepsize()

        for _ in range(self.max_iter):
            # gradient descent step. Use of y_k instead of x
            grad = A.T @ (A @ y_k - b) # TODO: epxlain in report computations and formulations    
            
            # calculate new x with proximal operator
            x_new = self.regulirizer.prox_op(y_k, t_k, grad)
            t_k_new = (1+ np.sqrt(1 + 4 * t_k**2) ) / 2  # Update t for k+1
            y_new = x_new + ( (t_k - 1) / t_k_new) * (x_new - x)  # Update y for k+1

            # compute loss (total loss = data loss + reg loss)
            loss = 0.5 * np.linalg.norm(A @ x_new - b)**2 + self.regulirizer.compute_reg_loss(x)            
            self.loss_history.append(loss)
            
            # stop criterion
            if self._check_convergence(x, x_new): break
            x = x_new
            y_k = y_new
            t_k = t_k_new
        
        return x
