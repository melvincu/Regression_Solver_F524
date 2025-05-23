import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from problems import Lasso, Ridge, ElasticNet
from algorithms import ProxGradient, FISTA, BFGS, DFP
from common import get_data

def main():
    # ----- data -----
    X_train, X_test, y_train, y_test = get_data()
    
    # ----- solve -----
    problem = Ridge(lbd=1)
    algo = BFGS(max_iter=1000)
    w_pred = algo.solve(problem, X_train, y_train, verbose=True)
    
    #  ----- results -----
    print(f"[DEBUG] w_pred{w_pred.shape} = {w_pred}")
    y_pred_train = X_train@w_pred
    y_pred_test = X_test@w_pred
    print("[DEBUG] train MSE:", mean_squared_error(y_train, y_pred_train))
    print("[DEBUG] test MSE:", mean_squared_error(y_test, y_pred_test))

if __name__ == '__main__':
    main()

"""
TODO:
    - gradient computation optimization in BFGS and DFP (avoid double new grad computation)        
"""