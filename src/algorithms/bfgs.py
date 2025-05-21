import numpy as np
from .opti_algorithm import OptiAlgorithm
from problems.composite_prob import CompositeProblem

class BFGS(OptiAlgorithm):
    
    def update_H(self, H, s, y):
        '''
        chap3 formula 3.25
        '''
        
        s = s.reshape(-1, 1) # (n,1), to column vector (for (n,n)@(n,1))
        y = y.reshape(-1, 1) # (n,1), to column vector
        H_new = H - ((H @ s @ s.T @ H) / (s.T @ H @ s)) + ((y @ y.T) / (y.T @ s))
        return H_new 
    
    def update_B(self, B, s, y):
        '''
        chap3 formula 3.26
        '''
        
        n = B.shape[0]
        s = s.reshape(-1, 1) # (n,1), to column vector (for (n,n)@(n,1))
        y = y.reshape(-1, 1) # (n,1), to column vector        
        ratio = 1/(y.T@s)
        l_part = np.eye(n) - (ratio * (s @ y.T)) 
        r_part = np.eye(n) - (ratio * (y @ s.T))
        B_new =  l_part@B@r_part + (ratio * (s @ s.T)) 
        return B_new
    
    # Note: can also do binary search on linspace [0,1] see (https://medium.com/gaussian-machine/implement-l-bfgs-optimization-from-scratch-3e5c909f9079)
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
        B = np.eye(n)  # B0 (can also do Hessian approx H, can do smarter initializations too)
        
        for iter in range(self.max_iter):
            
            # ----- search direction ----- 
            grad = problem.g_gradient(A, b, w)
            
            d = -B @ grad # (if use H then -np.linalg.inv(H)@grad)
            
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
            B = self.update_B(B, s, y) # (n,n)
            w = w_new

        if verbose: print(f"(Solved in {iter} iterations")

        return w