import numpy as np

from problems.composite_prob import CompositeProblem
from .opti_algorithm import OptiAlgorithm

# (FISTA)
class FISTA(OptiAlgorithm):
    def __init__(self, problem:CompositeProblem):
        super().__init__()
        self.problem = problem
        
    def solve(self, A, b, verbose=False):
        n = A.shape[1]      # num features
        x = np.zeros((n,1)) # w0 (nx1)
        L = self.compute_lipschitz_const(A)
        step = 1/L
        
        y = np.zeros((n,1)) # y0 (nx1)
        t = 1
        
        for iter in range(self.max_iter):
                
            # ----- proximal gradient step -----
            g_grad = self.problem.g_gradient(y)        
            x_new = self.problem.h_proximal_op(y-step*g_grad,step)
        
            # ----- update momentum -----
            t_new = 0.5*(1+np.sqrt(1 + 4 * t**2))
            y_new = x_new + ((t-1)/t_new)*(x_new-x)
            
            # ----- loss -----
            loss = self.problem.obj_value(x) # residual
            self.loss_history.append(loss)
            
             # ----- check convergence -----
            if (self.has_converged(x, x_new)): break
            
            # ----- update -----
            x = x_new
            y = y_new
            t = t_new
                            
        if verbose: print(f"(Solved in {iter} iterations")
        return x