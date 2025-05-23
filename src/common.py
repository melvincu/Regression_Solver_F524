from dataclasses import dataclass
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# @dataclass
# class AlgoStats():
#     regul:int
#     it_num:int
#     loss_hist:list[float]
#     t_grad:float
#     t_proxop:float
#     t_tot:float
#     memory:float
    
def get_data():
    df = fetch_california_housing()
    X = df.data
    y = df.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test # X(m,n), y(m,)
