import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes

from line_search import * # TODO: abstraction

from proximal_operators import L1Prox, ElasticNetProx
from algorithms import ISTA


def main():
    # ----- Data -----
    data = load_diabetes()
    X, y  = data.data, data.target
    
    #############################################################
    # eg. fixed stepsize ISTA for elastic-net regression 
    #############################################################
    
    line_search = FixedStepSize(X)
    prox_op = ElasticNetProx(0.1, 0.1)
    solver = ISTA(line_search, prox_op)
    
    w_ista = solver.solve(X, y)
    print(f"ISTA sol = {w_ista}")
    
if __name__ == '__main__':
    main()
    

"""
TODO:
    - load data / find regr dataset
    - line search abstraction / decouple
    - proximal operator not required for L-BFGS algorithm (optional param of BaseSolver ??)
    - check proximal step params !!!
    - do correct ista impl -> check work -> extend solver with other algos
    - do readme
"""