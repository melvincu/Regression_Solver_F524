import numpy as np
import time

from .opti_algorithm import OptiAlgorithm
from problems.composite_prob import CompositeProblem

class FISTA(OptiAlgorithm):
  
    def solve(self, problem:CompositeProblem, A, b, verbose=False):
        m,n = A.shape # n_samples, n_features
        w = np.zeros(n) # (n,) != (nx1)
        y = np.zeros(n) # y0
        
        L = self.compute_lipschitz_const(A)
        step = 1.0/L
        
        t = 1
        
        # Timers
        t_grad_total = 0.0
        t_prox_total = 0.0
        t_start = time.perf_counter()
        
        for iter in range(self.max_iter):
                
            # ----- proximal gradient step -----
            t0 = time.perf_counter()
            g_grad = problem.g_gradient(A, b, y)        
            t_grad_total += time.perf_counter() - t0

            t0 = time.perf_counter()
            w_new = problem.h_proximal_op(y-step*g_grad,step)
            t_prox_total += time.perf_counter() - t0

            # ----- update momentum -----
            t_new = 0.5*(1+np.sqrt(1 + 4 * t**2))
            y_new = w_new + ((t-1)/t_new)*(w_new-w)
            
            # loss history
            loss = problem.obj_value(A, b, w_new)
            self.loss_history.append(loss)
            
             # ----- check convergence -----
            if (self.has_converged(w, w_new)): 
                self.iter_num = iter
                break
            
            # ----- update -----
            w = w_new
            y = y_new
            t = t_new
                            
        t_total = time.perf_counter() - t_start

        stats = {"t_grad": t_grad_total, "t_prox": t_prox_total, "t_tot": t_total, "it_num": self.iter_num}
        
        return w, stats