import numpy as np

from problems.composite_prob import CompositeProblem
from .opti_algorithm import OptiAlgorithm

# (FISTA)
class FISTA(OptiAlgorithm):
    def __init__(self, problem:CompositeProblem):
        super().__init__()
        self.problem = problem
        
    def solve(self, A, b, verbose=False):
        m,n = A.shape # n_samples, n_features
        w = np.zeros(n) # (n,) != (nx1)
        y = np.zeros(n) # y0
        
        L = self.compute_lipschitz_const(A)
        step = 1.0/L
        
        t = 1
                
        for iter in range(self.max_iter):
                
            # ----- proximal gradient step -----
            g_grad = self.problem.g_gradient(y)        
            w_new = self.problem.h_proximal_op(y-step*g_grad,step)
        
            # ----- update momentum -----
            t_new = 0.5*(1+np.sqrt(1 + 4 * t**2))
            y_new = w_new + ((t-1)/t_new)*(w_new-w)
            
            # loss history
            loss = self.problem.obj_value(w_new)
            self.loss_history.append(loss)
            
             # ----- check convergence -----
            if (self.has_converged(w, w_new)): break
            
            # ----- update -----
            w = w_new
            y = y_new
            t = t_new
                            
        if verbose: print(f"(Solved in {iter} iterations")
        
        return w