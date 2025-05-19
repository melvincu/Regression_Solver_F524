import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from problems import Lasso, ElasticNet, OLS
from algorithms import ProxGradient, FISTA, LBFGS
from experiments.experiments import make_experiments
   
def get_data():
        df = fetch_california_housing()

        # Get features and target
        X = df.data
        y = df.target
        
        # train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
            
        return X_train, X_test, y_train, y_test # X(m,n), y(m,)

def main():

    X_train, X_test, y_train, y_test = get_data()
    df = make_experiments(X_train, y_train)
    print(df.head())
    
    # # solve
    # problem = Lasso(lbd=1)
    # algo = ProxGradient(problem)
    # w_ista = algo.solve(X_train, y_train, verbose=True)
    
    # # results
    # y_pred_train = X_train@w_ista
    # y_pred_test = X_test@w_ista
    # print("train MSE:", mean_squared_error(y_train, y_pred_train))
    # print("test MSE:", mean_squared_error(y_test, y_pred_test))
    
    
if __name__ == '__main__':
    main()

"""
TODO:
    - code structure refractor (data loading and experiments)
    - display results stats/properties (datframe.describe)
    - plot loss history(iter vs loss/obj), sparsity(lbd vs sparisty), mse (lbd vs mse) for different lambda (for all problem/algo combination ?)
    - backtracking line search
    - add at least one other problem (glm, logistic, svm, ...) ?
    - add L-BFGS algorithm
"""