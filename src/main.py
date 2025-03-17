import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes

from line_search import * # TODO: abstraction

from regulirizers import L1Regulirizer, ElasticNetRegulirizer
from algorithms import ISTA, GradientDesc


def main():
    # --------------- Data ---------------
    # data = load_diabetes()
    # X, y  = data.data, data.target
    
    # --------------- Synth Data ---------------
    np.random.seed(42)
    m, n  = 100, 3
    
    X = np.random.randn(m, n) # features
    B = np.array([[2], [-3], [1]])  # True weights
    w = np.random.randn(m, 1) * 0.5  # noise
    
    # (reconstruct target y based on samples X, weights B and rand noise w)
    y = X @ B + w # y = XB + w 
    
    #############################################################
    # eg. ISTA with fixed stepsize on l1-regression 
    #############################################################
    
    line_search = FixedStepSize(X)
    prox_op = L1Regulirizer(0.1)
    solver = ISTA(line_search, prox_op)    
    # solver = GradientDesc(line_search, prox_op)
    
    ista_B = solver.solve(X, y)
    
    print(f"ISTA_B = {ista_B}")
    print(f"B = {B}")
    
if __name__ == '__main__':
    main()
    
"""
TODO:
    - load data / find regr dataset
    - do l2 and elastic net regulirizer proximal operators correctly
    - add line search to solve methods - currently just do cst step size 0.01 (abstraction too)
    - do readme
"""