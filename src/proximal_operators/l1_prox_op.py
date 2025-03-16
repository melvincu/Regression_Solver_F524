import numpy as np

from .base_prox_op import BaseProxOp

class L1Prox(BaseProxOp):
    """
    l1-norm proximal operator
    
    params:
        - lbd
    """

    def __init__(self, lbd):
        super().__init__()
        self.lbd =lbd
        
    def apply(self, x):
        return np.sign(x) * np.maximum(np.abs(x) - self.lbd, 0)
