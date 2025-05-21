import numpy as np
from .opti_algorithm import OptiAlgorithm
from problems.composite_prob import CompositeProblem

class BFGS(OptiAlgorithm):
    
    def bfgs_update(self, H, s, y):
        s = s.reshape(-1, 1) # (n,1), to column vector (for (n,n)@(n,1))
        y = y.reshape(-1, 1) # (n,1), to column vector
        H_new = H - (H @ s @ s.T @ H) / (s.T @ H @ s) + (y @ y.T) / (y.T @ s)
        return H_new            
    
    # Note: can do binary search on linspace [0,1] see (https://medium.com/gaussian-machine/implement-l-bfgs-optimization-from-scratch-3e5c909f9079)
    def wolfe_line_search(self, problem:CompositeProblem, A, b, w, d, grad, alpha=0.5):
        t = 1
        c1 = 1e-4 
        c2 = 0.9 
        fw = problem.obj_value(A, b, w) 
        w_new = w + t*d 
        grad_new = problem.g_gradient(A, b, w_new)
        
        # until both wolf conditions repsected (strong, fro weak conditions just remove np.abs in curvature cond)
        while problem.obj_value(A, b, w_new) >= fw + (c1*t*grad.T@d) or np.abs(grad_new.T@d) <= np.abs(c2*grad.T@d): 
            t *= alpha
            w_new = w + t*d 
            grad_new = problem.g_gradient(A, b, w_new)

        return t
    
    def solve(self, problem:CompositeProblem, A, b, verbose=False):
        m,n = A.shape # n_samples, n_features
        w = np.zeros(n) # w0 (n,)
        H = np.eye(n)  # H0 (Hessian approx - can do smarter initializations)
        
        for iter in range(self.max_iter):
            
            # ----- search direction ----- 
            grad = problem.g_gradient(A, b, w)
            d = -H@grad
            
            # ----- step -----
            t = self.wolfe_line_search(problem, A, b, w, d, grad)
            w_new = w + t*d
            
            # -----  loss history ----- 
            loss = problem.obj_value(A, b, w_new)
            self.loss_history.append(loss)
            
             # ----- check convergence -----
            if (self.has_converged(w, w_new)): break
            # if (np.linalg.norm(grad) < self.eps): break
            
            # ----- update ----- 
            grad_new = problem.g_gradient(A, b, w_new)
            s = w_new - w # (n,)
            y =  grad_new - grad # (n,)
            H = self.bfgs_update(H, s, y) # (n,n)
            w = w_new

        if verbose: print(f"(Solved in {iter} iterations")

        return w