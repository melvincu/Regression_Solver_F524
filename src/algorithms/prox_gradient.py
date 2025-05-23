import numpy as np
import time

from problems.composite_prob import CompositeProblem
from .opti_algorithm import OptiAlgorithm

# (FISTA)
class ProxGradient(OptiAlgorithm):
 
    def solve(self, problem:CompositeProblem, A, b, verbose=False):
        m,n = A.shape # n_samples, n_features
        w = np.zeros(n) # (n,) != (nx1) ()
        
        L = self.compute_lipschitz_const(A)
        step = 1.0/L
        
        # Timers
        t_grad_total = 0.0
        t_prox_total = 0.0
        t_start = time.perf_counter()
        
        for iter in range(self.max_iter):
            
            # ----- proximal gradient step -----
            t0 = time.perf_counter()
            g_grad = problem.g_gradient(A, b, w)
            t_grad_total += time.perf_counter() - t0
            
            t0 = time.perf_counter()
            w_new = problem.h_proximal_op(w-step*g_grad,step)
            t_prox_total += time.perf_counter() - t0

            # loss history
            loss = problem.obj_value(A, b, w_new)
            self.loss_history.append(loss)
            
            # ----- check convergence -----
            if (self.has_converged(w, w_new)):
                self.iter_num = iter
                break
                        
            # ----- update -----
            w = w_new

        t_total = time.perf_counter() - t_start

        stats = {"t_grad": t_grad_total, "t_prox": t_prox_total, "t_tot": t_total, "it_num": self.iter_num}
        
        return w, stats