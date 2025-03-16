import numpy as np

from .base_prox_op import BaseProxOp

class L2Prox(BaseProxOp):
    """
    l1-norm proximal operator
    
    params:
        - lbd
    """
        
    def __init__(self, lbd):
        super().__init__()
        self.lbd = lbd
        
    def apply(self, x):
        return x / (1 + 2 * self.ldb)