import numpy as np

from problems.composite_prob import CompositeProblem
from .opti_algorithm import OptiAlgorithm

# (FISTA)
class ProxGradient(OptiAlgorithm):
    def __init__(self, problem:CompositeProblem):
        super().__init__()
        self.problem = problem
        
    def solve(self, A, b, verbose=False):    
        m,n = A.shape # n_samples, n_features
        w = np.zeros(n) # (n,) != (nx1)
        
        L = self.compute_lipschitz_const(A)
        step = 1.0/L
        
        for iter in range(self.max_iter):
            
            # ----- proximal gradient step -----
            g_grad = self.problem.g_gradient(w)
            w_new = self.problem.h_proximal_op(w-step*g_grad,step)

            # ----- check convergence -----
            if (self.has_converged(w, w_new)): break
                        
            # ----- update -----
            w = w_new

        if verbose: print(f"({iter} iterations)")

        return w