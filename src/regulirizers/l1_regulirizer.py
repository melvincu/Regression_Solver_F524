import numpy as np

from .regularizer import Regularizer

class L1Regulirizer(Regularizer):
    """
    (l1) Lasso regulirizer
    
    params:
        - lbd
    """

    def __init__(self, lbd):
        super().__init__()
        self.lbd =lbd
        
    def prox_op(self, x, t, grad):
        """
        shrink(x-t*(A.T*(Ax-b)))
        """
        _x = x-t*grad
        return np.sign(_x) * np.maximum(np.abs(_x) - t*self.lbd, 0)

    def compute_reg_loss(self, x):
        """
        lbd * ||x||_1
        """
        return self.lbd * np.linalg.norm(x,ord=1) # np.sum(np.abs(x))
