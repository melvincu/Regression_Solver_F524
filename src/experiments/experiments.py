import pandas as pd

from problems import Lasso, Ridge, ElasticNet
from algorithms import ProxGradient, FISTA, BFGS

def make_experiments(X_train, y_train):
        
    ex_stats = {
        "problem": "lasso",
        "algo": "ista",
        "lambda": 1,
        "iterations": 400,
        "train_mse": 5,
        "test_mse": 10,
        "grad_time": 0.06,
        "prox_time": 0.05,
        "total_time": 0.012
        }
    
    results_list = []
    reguls = [0.1, 1, 10, 100]
    problems = ["lasso", "elasticnet", "ols"]
    for problem in problems:
        for lbd in reguls:
            
            p = None
            match(problem):
                case "lasso":
                    p = Lasso(lbd=lbd)
                case "elasticnet":
                    p = ElasticNet(lbd1=lbd, lbd2=lbd)
                case "ols":
                    p = OLS()
                case _:
                    raise ValueError(f"[experiments] {problem} not recognized")
            
            algo = ProxGradient(problem=p)
            
            # [algo] on [problem] with [regul]
            run_stats = algo.solve(X_train, y_train)
            results_list.append(ex_stats)
            
    results_df = pd.DataFrame(results_list)
    return results_df



"""
1) Summary stats
    DataFrame.groupby(["problem", "algo", "lambda"]).describe()
    to get mean, min, max, std for execution time and iterations

2) algorithm test mse comparison:
    compare algo test mse on same problems, so a [bar plot] for each problem with test mse for each regul level (ticks)
    e.g. for lasso, x_tick = (lbd=1), 3 bars showing test mse (1 for each algo)
      
3) algorithm convergence comparison:
    compare algo convergences on same problems, so a [bar plot] for each problem with convergence (num iter) for each regul level (ticks)
    e.g. for lasso, x_tick = (lbd=1), 3 bars showing convergence (1 for each algo)
"""