import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from problems import Lasso, Ridge, ElasticNet
from algorithms import ProxGradient, FISTA, BFGS, DFP
from experiments.experiments import make_experiments

def get_data():
    
    df = fetch_california_housing()
    
    X = df.data
    y = df.target
    
    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test # X(m,n), y(m,)

def main():

    X_train, X_test, y_train, y_test = get_data()
    # df = make_experiments(X_train, y_train)
    # print(df.head())
    
    # solve
    problem = Ridge(lbd=1)
    algo = BFGS(max_iter=1000)
    w_pred = algo.solve(problem, X_train, y_train, verbose=True)
    
    # results
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
        
    - plot loss history(iter vs loss/obj), sparsity(lbd vs sparisty), mse (lbd vs mse) for different lambda (for all problem/algo combination ?)
"""