import numpy as np

from problems.composite_prob import CompositeProblem
from .opti_algorithm import OptiAlgorithm, timeit

# (FISTA)
class ProxGradient(OptiAlgorithm):
 
    @timeit    
    def solve(self, problem:CompositeProblem, A, b, verbose=False):
        m,n = A.shape # n_samples, n_features
        w = np.zeros(n) # (n,) != (nx1) ()
        
        L = self.compute_lipschitz_const(A)
        step = 1.0/L
        
        for iter in range(self.max_iter):
            
            # ----- proximal gradient step -----
            g_grad = problem.g_gradient(A, b, w)
            w_new = problem.h_proximal_op(w-step*g_grad,step)

            # loss history
            loss = problem.obj_value(A, b, w_new)
            self.loss_history.append(loss)
            
            # ----- check convergence -----
            if (self.has_converged(w, w_new)): 
                self.s_iter = iter
                break
                        
            # ----- update -----
            w = w_new

        if verbose: print(f"({iter} iterations)")

        res = {"[ProxGrad] tot_t":self.s_time, "it_num":self.s_iter, "w":w}
        return w