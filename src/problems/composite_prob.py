from abc import ABC, abstractmethod

# convex
class CompositeProblem(ABC):
    
    @abstractmethod
    def g_gradient(self, x):
        """gradient of smooth function g in x"""
        pass
    
    @abstractmethod
    def h_proximal_op(self, x, t):
        "proximal operator of non smooth function h in x"
        pass

    def obj_value(self, x):
        """objective function value in x"""
        pass