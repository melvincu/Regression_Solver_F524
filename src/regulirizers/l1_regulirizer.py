import numpy as np

from .regularizer import Regularizer

class L1Regulirizer(Regularizer):
    """
    l1-norm regulirizer
    
    params:
        - lbd
    """

    def __init__(self, lbd):
        super().__init__()
        self.lbd =lbd
        
    def prox_op(self, x, t):
        return np.sign(x) * np.maximum(np.abs(x) - t*self.lbd, 0)

    def compute_reg_loss(self, x):
        return self.lbd * np.sum(np.abs(x))
