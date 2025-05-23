import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from problems import CompositeProblem, Lasso, Ridge, ElasticNet
from algorithms import OptiAlgorithm, ProxGradient, FISTA, BFGS
from common import get_data

# algos noy handling non smooth parts    
smooth_algos:dict[str, OptiAlgorithm] = {"prox_grad": ProxGradient(),
                                            "fista": FISTA(),
                                            "bfgs": BFGS()}
# algos handling both
n_smooth_algos:dict[str, OptiAlgorithm] = {"prox_grad": ProxGradient(),
                                            "fista": FISTA()}

# smooth problems
smooth_problems:dict[str, CompositeProblem] = {"ridge": Ridge}

# others
n_smooth_problems:dict[str, CompositeProblem] = {"lasso":Lasso,
                                                 "ridge":Ridge,
                                                 "enet": ElasticNet}


def plot_problem_algo_convergence(problem:CompositeProblem, algos:dict[str,OptiAlgorithm]):
    '''
    Plot loss function vs iterations for different algos on same problem (maybe do multiple regul params too)
    '''

    X_train, X_test, y_train, y_test = get_data()

    plt.figure(figsize=(8,5))

    for name, algo in algos.items():
        w = algo.solve(problem, X_train, y_train)
        plt.plot(algo.loss_history, label=name)

    plt.xlabel("Iteration")
    plt.ylabel("Objective / Loss")
    plt.title(f"[ProblemName] Convergence (iteration vs loss) for different solvers")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# def df_to_tex(df:pd.DataFrame):
#     latex_table = df.to_latex(
#     )

#     with open('results_table.tex', 'w') as f:
#         f.write(latex_table)

def measurement_summary(df: pd.DataFrame) -> pd.DataFrame:
    # aggregate the stats for each measurement column
    
    # stats features columns
    columns = df.columns.drop(['problem', 'regul']) 
    # columns = df.columns.drop(['algo', 'regul']) 

    agg_stats = df.groupby(["problem", "regul"]).agg({
        col: ["min", "max", "mean", "std"]
        for col in columns
    })
    
    # flatten
    agg_stats.columns = ["_".join(col) for col in agg_stats.columns]
    return agg_stats

def measurements_dataframe(R:int, algo: OptiAlgorithm, problems:dict[str,CompositeProblem]):
    '''
    gather stats of algo over R runs
    '''
    
    X_train, X_test, y_train, y_test = get_data()
    
    lbd_grid = [0.1, 1, 10]
    
    records = []
    for problem_name, Problem in problems.items():
        for lbd_setting in lbd_grid:
            for k in range(R):
                
                problem = None
                if problem_name == "enet":
                    problem = Problem(lbd_setting, lbd_setting)
                else:
                    problem = Problem(lbd_setting)
                
                w, stats = algo.solve(problem, X_train, y_train)
            
                records.append({"problem": problem_name,
                                "regul": lbd_setting,
                                **stats
                                })
                    
    df = pd.DataFrame(records)
    
    return df

def solutions_dataframe(Problem: CompositeProblem, algos:dict[str,OptiAlgorithm]):
    '''
    gather solution quality of algos on problem
    '''
    
    X_train, X_test, y_train, y_test = get_data()
    
    lbd_grid = [0.1, 1, 10]
    
    records = []
    for algo_name, algo in algos.items():
        for lbd_setting in lbd_grid:                
                problem = None
                
                if Problem == ElasticNet:
                    problem = Problem(lbd_setting, lbd_setting)
                else:
                    problem = Problem(lbd_setting)
                
                w, stats = algo.solve(problem, X_train, y_train)
                
                # sol quality
                y_pred_train = X_train@w
                y_pred_test = X_test@w
                train_mse = mean_squared_error(y_train, y_pred_train)
                test_mse = mean_squared_error(y_train, y_pred_train)
                
                records.append({"algo": algo_name,
                                "regul": lbd_setting,
                                "train_mse": train_mse,
                                "test_mse": test_mse
                                })
                    
    df = pd.DataFrame(records)
    
    return df

def experiments():
    X_train, X_test, y_train, y_test = get_data()
        
    # -------------
    # plot_problem_algo_convergence(Ridge(lbd=1), smooth_algos)
    # plot_problem_algo_convergence(Lasso(lbd=1), n_smooth_algos)
    # plot_problem_algo_convergence(ElasticNet(lbd1=1,lbd2=1), n_smooth_algos)
    
    # -------------
    # pg_stats_df = measurements_dataframe(5, ProxGradient(), n_smooth_problems)
    # pg_df_summary = measurement_summary(pg_stats_df)
    
    # fista_stats_df = measurements_dataframe(5, FISTA(), n_smooth_problems)
    # fista_df_summary = measurement_summary(fista_stats_df)
    
    # bfgs_stats_df = measurements_dataframe(5, BFGS(), smooth_problems)
    # bfgs_df_summary = measurement_summary(bfgs_stats_df)
    
    # res = fista_df_summary.T
    # print(res)
    
    # ------------
    # ridge_sol_df = solutions_dataframe(Ridge, smooth_algos)
    # lasso_sol_df = solutions_dataframe(Lasso, n_smooth_algos)
    # enet_sol_df = solutions_dataframe(ElasticNet, n_smooth_algos)

    # print(enet_sol_df.T)
    
if __name__ == '__main__':
    experiments()
