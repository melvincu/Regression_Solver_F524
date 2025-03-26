import numpy as np

from problems.composite_prob import CompositeProblem
from .opti_algorithm import OptiAlgorithm

# (FISTA)
class ProxGradient(OptiAlgorithm):
    def __init__(self, problem:CompositeProblem):
        super().__init__()
        self.problem = problem
        
    def solve(self, A, b):
        n = A.shape[1]      # num features
        x = np.zeros((n,1)) # w0 (nx1)
        
        for _ in range(self.max_iter):
            t = self.fixed_stepsize(A)

            # step
            grad = self.problem.gradient(x)
            x_new = self.problem.proximal_op(x-t*grad,t)
            
            # loss
            # loss = problem.value(x) - true # TODO (MSE?)
            # self.loss_history.append(loss)
            
            # stop criterion
            if self.check_convergence(x, x_new): break
            x = x_new

        return x