import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from data_processor import get_data
from problems import Lasso, ElasticNet
from algorithms import ProxGradient, FISTA


def main():
    
    # --------------- Dataset ---------------
    print("Loading data ....")     
    X_train, X_test, y_train, y_test = get_data() # X(m,n), y(m,)
    
    print(y_train.shape)
    exit(0)
    #############################################################
    # eg. [ISTA] with [fixed stepsize] on [l1-regularization] least squares regression
    #############################################################

    print("Solving regression ....")     
    problem = Lasso(X_train, y_train, lbd=1)
    # algo = ProxGradient(problem)
    algo = FISTA(problem)

    w_ista = algo.solve(X_train, y_train, verbose=True)

    # results
    y_pred_train = X_train@w_ista
    y_pred_test = X_test@w_ista
    print("train MSE:", mean_squared_error(y_train, y_pred_train))
    print("test MSE:", mean_squared_error(y_test, y_pred_test))

    
if __name__ == '__main__':
    main()


"""
TODO:
    - loss history, plots, stats, .... (specific class ? ploting ...)
    - backtracking line search
    - add at least one other problem (glm, logistic, svm, ...) ?
"""