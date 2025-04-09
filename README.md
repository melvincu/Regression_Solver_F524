# Continuous Optimization: Regression solver

### Course

+ course: INFO-F524
+ year: MA2

### Description
Modular Regression solver handling general composite optimization problems (proximal gradient), extendable to more optimization problems their solving algorithms (extend abstract classes).

Problems currently supported:
+ LASSO
+ Elastic-Net

Solving algorithms currently implemented:
+ ISTA
+ FISTA
+ L-BFGS

### Usage
```
problem = Problem(X_train, y_train, ...)
algo = Algo(problem)
w_pred = algo.solve(X_train, y_train)

y_pred_test = X_test @ w_ista
print("test MSE:", mean_squared_error(y_test, y_pred_test))
```