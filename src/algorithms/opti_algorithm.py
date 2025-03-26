from abc import ABC, abstractmethod
import numpy as np

class OptiAlgorithm(ABC):
    
    def __init__(self, max_iter=1000, eps=1e-5):
        self.max_iter = max_iter
        self.eps = eps
        self.loss_history = []
    
    def get_loss_histroy(self):
        return self.loss_history
    
    def check_convergence(self, x, x_new):
        return (np.linalg.norm(x_new - x) < self.eps)    
    
    def fixed_stepsize(self, A):
        return 1/(np.linalg.norm(A, ord=2)**2) # 1/L

    def backtracking_stepsize(self, A, b, x, t, grad):
            # t = 0.1 # t0
            
            # # f(x)
            # f = 0.5 * np.linalg.norm(A @ x - b)**2 + regulirizer.compute_reg_loss(x)
            
            # # f(x - t*grad)
            # _x = x-t*grad
            # f_step = 0.5 * np.linalg.norm(A @ _x - b)**2 + self.regulirizer.compute_reg_loss(_x)

            # # f(x - t*grad) - f(x) < c*t*||grad||_2^2
            # while f_step - f < self.c * t * np.linalg.norm(grad)**2:
            #     t *= self.alpha
            # return t
            pass
        
    @abstractmethod
    def solve(self):
        pass
    
    