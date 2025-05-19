from abc import ABC, abstractmethod

# convex
class CompositeProblem(ABC):
    
    @abstractmethod
    def g_gradient(self, X, y, w):
        """gradient of smooth function g in w"""
        pass
    
    @abstractmethod
    def h_proximal_op(self, w, t):
        "proximal operator of non smooth function h in w"
        pass

    def obj_value(self, X, y, w):
        """objective function value in w"""
        pass