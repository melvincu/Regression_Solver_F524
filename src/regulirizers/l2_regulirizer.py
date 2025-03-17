import numpy as np

from .regularizer import Regularizer

class L2Regulirizer(Regularizer):
    """
    l1-norm proximal operator
    
    params:
        - lbd
    """
        
    def __init__(self, lbd):
        super().__init__()
        self.lbd = lbd
        
    def prox_op(self, x, t):
        return x / (1 + 2 * t*self.lbd)
    
    def compute_reg_loss(self, x):
        return 1