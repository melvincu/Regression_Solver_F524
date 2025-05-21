import numpy as np
from .opti_algorithm import OptiAlgorithm
from problems.composite_prob import CompositeProblem

class DFP(OptiAlgorithm):
    
    def dfp_update(self, B, s, y):
        s = s.reshape(-1, 1) # (n,1), to column vector (for (n,n)@(n,1))
        y = y.reshape(-1, 1) # (n,1), to column vector
        B_new = B - (B @ y @ y.T @ B) / (y.T @ B @ y) + (s @ s.T) / (y.T @ s)

        return B_new            
    
    def wolfe_line_search(self, problem:CompositeProblem, A, b, w, d, grad, alpha=0.5):
        
        t = 1
        c1 = 1e-4 
        c2 = 0.9 
        fw = problem.obj_value(A, b, w) 
        w_new = w + t*d 
        grad_new = problem.g_gradient(A, b, w_new)
        
        # until both wolf conditions repsected
        while problem.obj_value(A, b, w_new) >= fw + (c1*t*grad.T@d) or grad_new.T@d <= c2*grad.T@d: 
            t *= alpha
            w_new = w + t*d 
            grad_new = problem.g_gradient(A, b, w_new)
            
        return t
    
    def solve(self, problem:CompositeProblem, A, b, verbose=False):
        m,n = A.shape # n_samples, n_features
        w = np.zeros(n) # w0 (n,)
        B = np.eye(n)  # B0 (hessian inv approx)
        
        for iter in range(self.max_iter):
            
            # ----- search direction ----- 
            grad = problem.g_gradient(A, b, w)
            d = -B@grad
            
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
            B = self.dfp_update(B, s, y) # (n,n)
            w = w_new

        if verbose: print(f"(Solved in {iter} iterations")

        return w