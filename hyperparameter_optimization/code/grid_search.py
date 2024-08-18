import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# load data
filename = 'diabetes.csv'
names = ['pregnancies', 'glucose', 'DBP', 'skinfold', 'insulin' ,'BMI' ,'diabetes_pedigree' ,'age' ,'outcome']
data = read_csv(filename, names=names)

# create array w/ data and seperate array into input/outputs
array = data.values # inputs = all rows in columns 0-7
x = array[:,0:8] # output = all rows in column 8
y= array[:,8]

# set model
model = LogisticRegression()

# select hyperparameters
solvers = ['newton-cg', 'lbfgs', 'liblinear', 'saga']
penalty = ['l2']
penalty_strength = [100, 10, 1.0, 0.1, 0.01]

# grid search parameter tuning
grid = dict(solver=solvers, penalty=penalty, C=penalty_strength)
grid = GridSearchCV(estimator=model, param_grid=grid, n_jobs=1, scoring='accuracy', error_score=0)
grid.fit(x,y)

# print results
print('Model Accuracy:', grid.best_score)
print('Best Solver:', grid.best_estimator.solver)
print('Best Penalty Strength:', grid.best_estimator_.C)
