# Continuous Optimization: Regression solver

### Course

+ course: INFO-F524
+ year: MA2

### Description
Modular Regression solver handling general composite optimization problems (proximal gradient), extendable to more optimization problems and their solving algorithms (extend abstract classes).

Problems currently supported (Least square based):
+ LASSO
+ Ridge
+ Elastic-Net

Solving algorithms currently implemented:
+ ISTA
+ FISTA
+ BFGS

### Usage
```
problem = Problem()
algo = Algo(problem)

w_pred = algo.solve(X_train, y_train)
y_pred_test = X_test @ w_pred
```