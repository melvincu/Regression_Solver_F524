from abc import ABC, abstractmethod
import numpy as np

from problems.composite_prob import CompositeProblem

class OptiAlgorithm(ABC):
    
    def __init__(self, max_iter=1000, eps=1e-6):
        self.max_iter = max_iter
        self.eps = eps
        
        # solving stats
        self.loss_history = []
        self.iter_num = 0
    
    def loss_histroy(self):
        return self.loss_history
    
    def has_converged(self, x, x_new):
        return (np.linalg.norm(x_new - x) < self.eps) 
    
    def compute_lipschitz_const(self, A):
        return np.linalg.norm(A, ord=2)**2 # L
        
    @abstractmethod
    def solve(self, problem:CompositeProblem, A, b, verbose=False):
        pass

# def timeit(method):
#     def wrapper(self:OptiAlgorithm, *args, **kwargs):
#         start = time.time()
#         result = method(self, *args, **kwargs)
#         end = time.time()
#         self.s_time = end - start
#         return result
#     return wrapper
    
# def backtracking_stepsize(self, A, b, x, t, grad):
#     # ----- backtracking line search (on g) -----
#     # f_val = self.problem.obj_value(x)                               # f(x)
#     # f_step = self.problem.obj_value(x-t*grad)                       # f(x - t*grad)
#     # while (f_step - f_val) < (self.c*t*g_grad_norm*g_grad_norm):    # f(x - t*grad) - f(x) < c*t*||grad||_2^2
#     #     t *= self.alpha