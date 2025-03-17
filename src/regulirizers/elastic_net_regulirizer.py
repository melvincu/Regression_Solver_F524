import numpy as np

from .regularizer import Regularizer
from .l1_regulirizer import L1Regulirizer
from .l2_regulirizer import L2Regulirizer

class ElasticNetRegulirizer(Regularizer):
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
        
    def prox_op(self, x, t):
        # TODO: initialize L1 and L2 operator objects directly (in init) instead of creating them eveyr algo iteration !!
        return L2Regulirizer(self.lbd2).prox_op(x, t) * L1Regulirizer(self.lbd1).prox_op(x, t)
    
    def compute_reg_loss(self, x):
        return 1
