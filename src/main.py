import numpy as np

from problems import Lasso, ElasticNet
from algorithms import ProxGradient

def main():

    # --------------- Synth Data ---------------
    np.random.seed(42)
    m, n  = 100, 3 # 100 samples, 3 features
    
    A = np.random.randn(m, n) # features
    x = np.array([[2], [-3], [1]]) # True weights (minimizer coord)
    noise = np.random.randn(m, 1) * 0.5 # noise
    b = A @ x + noise # (construct target b based on samples A, true weights x and rand noise)
    
    #############################################################
    # eg. [ISTA] with [fixed stepsize] on [l1-regularization] least squares regression
    #############################################################
    
    problem = Lasso(A, b, lbd=0.1)
    algo = ProxGradient(problem)
    x_hat = algo.solve(A, b)
    
    print(f"x_algo = {x_hat.flatten()}")
    print(f"x_true = {x.flatten()}")

if __name__ == '__main__':
    main()