import numpy as np

from .base_prox_op import BaseProxOp
from .l1_prox_op import L1Prox
from l2_prox_op import L2Prox

class ElasticNetProx(BaseProxOp):
    """
    (l1 + l2)-norm proximal operator
    
    params:
        - lbd1
        - lbd2
    """

    def __init__(self, lbd1, lbd2):
        super().__init__()
        self.lbd1 = lbd1
        self.lbd2 = lbd2
        
    def apply(self, x):
        # TODO: initialize L1 and L2 operator objects directly (in init) instead of creating them eveyr algo iteration !!
        return L2Prox(self.lbd2).apply(x) * L1Prox(self.lbd1).apply(x)
