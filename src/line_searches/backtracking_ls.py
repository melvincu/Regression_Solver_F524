from .line_search import LineSearch

class BacktrackingLS(LineSearch):
    """Backtracking line search method (Armijo rule)."""
    
    def __init__(self, alpha, c):
        self.alpha = alpha  # Armijo parameter (sufficient decrease)
        self.c = c    # Reduction factor
    
    def compute_stepsize(self, X, x) -> int:
        # t = 1.0  # Start with a large step
        # while np.linalg.norm(X.T @ (X @ (w - t * grad) - y)) > self.alpha * np.linalg.norm(grad):
        #     t *= self.beta  # Reduce step size
        # return t
        pass