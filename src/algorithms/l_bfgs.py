import numpy as np

from problems.composite_prob import CompositeProblem
from .opti_algorithm import OptiAlgorithm, timeit

class LBFGS(OptiAlgorithm):
    def __init__(self, problem:CompositeProblem):
        super().__init__()
        self.problem = problem
    
    @timeit
    def solve(self, A, b, verbose=False):
        m,n = A.shape # n_samples, n_features
        w = np.zeros(n) # (n,) != (nx1)
        
        L = self.compute_lipschitz_const(A)
        step = 1.0/L
        
        pass
        
     