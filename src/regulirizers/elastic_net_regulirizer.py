import numpy as np

from .regularizer import Regularizer
from .l1_regulirizer import L1Regulirizer

class ElasticNetRegulirizer(Regularizer):
    """
    (l1-l2) elasticnet regulirizer
    
    params:
        - lbd1 (l1)
        - lbd2 (l2)
    """

    def __init__(self, lbd1, lbd2):
        super().__init__()
        self.lbd1 = lbd1
        self.lbd2 = lbd2
        
    def prox_op(self, x, t, grad):
        """
        shrink(x-t*(A.T*(Ax-b)+lbd2*x)) 
        
        (see chap2 p44)
        """
        n_grad = grad + self.lbd2*x
        return L1Regulirizer(self.lbd1).prox_op(x, t, n_grad) 
    
    def compute_reg_loss(self, x):
        """ lbd1 * ||x||_1 + (lbd2/2) * ||x||_2^2"""
        l1_loss = L1Regulirizer(self.lbd1).compute_reg_loss(x)
        l2_loss = 0.5 * self.lbd2 * np.sum(x ** 2)
        return l1_loss + l2_loss

# TODO: avoid creating new L1regulirizer object (member var ? or duplicate code ?)