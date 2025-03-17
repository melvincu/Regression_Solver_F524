from abc import ABC, abstractmethod

class Regularizer(ABC):
    """Base regulirizer abstract class"""
    
    @abstractmethod
    def prox_op(self, x, t):
        pass

    @abstractmethod
    def compute_reg_loss(self, x):
        pass