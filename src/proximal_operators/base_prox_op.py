from abc import ABC, abstractmethod

class BaseProxOp(ABC):
    """Base proximal operator abstract class"""
    
    @abstractmethod
    def apply(self, x):
        pass
