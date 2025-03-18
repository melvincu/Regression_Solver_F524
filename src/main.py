import numpy as np

from line_search import * # TODO: abstraction

from regulirizers import L1Regulirizer, ElasticNetRegulirizer
from algorithms import ISTA, GradientDesc

def main():
    # --------------- Data ---------------
    # data = load_diabetes()
    # X, y  = data.data, data.target
    
    # --------------- Synth Data ---------------
    np.random.seed(42)
    m, n  = 100, 3 # 100 samples, 3 features
    
    A = np.random.randn(m, n) # features
    w = np.array([[2], [-3], [1]])  # True weights (minimizer coord)
    noise = np.random.randn(m, 1) * 0.5  # noise
    
    # (reconstruct target b based on samples A, true weights w and rand noise)
    b = A @ w + noise
    
    #############################################################
    # eg. ISTA with fixed stepsize on l1-regression 
    #############################################################
    
    line_search = FixedStepSize(A)
    regulizer = L1Regulirizer(0.01)
    solver = ISTA(line_search, regulizer)    
    # solver = GradientDesc(line_search, prox_op)
    
    ista_w = solver.solve(A, b)
    
    print(f"ISTA w = {ista_w.reshape(1,n)[0]}")
    
if __name__ == '__main__':
    main()
    
"""
TODO:
    - data set selection and statistical properties - to check if dataset is relevant fro linear regression (features has some linear dependencies - to have good results)
    several dataset of increasing sizes/shape - imapct of n (num samples) / impact of m (num features) 
    
    - load data, do pipeline: load -> solve -> save measurements/plot!
    
    - add line search to solve methods - currently just do cst step size 0.01 (abstraction too)
    - do readme
    - add code to gather measurements: timer (maybe with decorators??), iteration convergence (gradient norm ?), loss (mse, ssr ?)
"""