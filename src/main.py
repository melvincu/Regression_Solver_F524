import numpy as np

from regulirizers import L1Regulirizer, ElasticNetRegulirizer
from algorithms import ISTA, GradientDesc
from line_searches import FixedStepSize, BacktrackingLS

def main():
    # --------------- Data ---------------
    # data = load_diabetes()
    # X, y  = data.data, data.target
    
    # --------------- Synth Data ---------------
    np.random.seed(42)
    m, n  = 100, 3 # 100 samples, 3 features
    
    A = np.random.randn(m, n) # features
    w = np.array([[2], [-3], [1]]) # True weights (minimizer coord)
    noise = np.random.randn(m, 1) * 0.5 # noise
    b = A @ w + noise # (reconstruct target b based on samples A, true weights w and rand noise)
    
    #############################################################
    # eg. [ISTA] with [fixed stepsize] on [l1-regularization] least squares regression
    #############################################################
    
    line_search = FixedStepSize(A)
    regulizer = L1Regulirizer(10)
    solver = ISTA(line_search, regulizer)    
    
    ista_w = solver.solve(A, b)
    
    print(f"ISTA w = {ista_w.reshape(1,n)[0]}")
    
if __name__ == '__main__':
    main()
    
"""
TODO:
    - data set selection and statistical properties - to check if dataset is relevant fro linear regression (features has some linear dependencies - to have good results)
    several dataset of increasing sizes/shape - imapct of n (num samples) / impact of m (num features) 
    
    - load data, do pipeline: load -> solve -> save measurements/plot!
    - do backtracking line search (need params unlike constant stepsize - problem with interface computestepszie method !)
    - do readme
    - add code to gather measurements: timer (maybe with decorators??), iteration convergence (gradient norm ?), loss (mse, ssr ?)
"""