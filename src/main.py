import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from data import KaggleDataHandler
from problems import Lasso, ElasticNet
from algorithms import ProxGradient, FISTA

def main():
    
    data_handler = KaggleDataHandler()
    X_train, X_test, y_train, y_test = data_handler.get_data() # X(m,n), y(m,)
    
    problem = Lasso(X_train, y_train, lbd=1)
    algo = ProxGradient(problem)
    w_ista = algo.solve(X_train, y_train, verbose=True)

    # --------------- Results ---------------
    y_pred_train = X_train@w_ista
    y_pred_test = X_test@w_ista
    print("train MSE:", mean_squared_error(y_train, y_pred_train))
    print("test MSE:", mean_squared_error(y_test, y_pred_test))

if __name__ == '__main__':
    main()

"""
TODO:
    - plot loss history(iter vs loss/obj), sparsity(lbd vs sparisty), mse (lbd vs mse) for different lambda (for all problem/algo combination ?)
    - backtracking line search
    - add at least one other problem (glm, logistic, svm, ...) ?
"""