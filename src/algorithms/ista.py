import numpy as np
from line_searches.line_search import *
from regulirizers import Regularizer
from .solver import Solver

class ISTA(Solver):
    def __init__(self, line_search:LineSearch , regulirizer:Regularizer, **kwargs):
        super().__init__(**kwargs)
        self.line_search = line_search
        self.regulirizer = regulirizer
    
    def solve(self, A, b):        
        n = A.shape[1] # num features
        x = np.zeros((n,1)) # w0 (nx1)

        for _ in range(self.max_iter):
            
            # compute step size t_k
            t_k = self.line_search.compute_stepsize()
            
            # gradient descent + shrinkage
            grad = A.T @ (A@x - b) # loss gradient (SSR)
            x_new = self.regulirizer.prox_op(x,t_k,grad)
            
            # total loss = SSR + reg loss
            loss = 0.5 * np.linalg.norm(A @ x_new - b)**2 + self.regulirizer.compute_reg_loss(x)            
            self.loss_history.append(loss)
            
            # stop criterion
            if self._check_convergence(x, x_new): break
            x = x_new

        return x
