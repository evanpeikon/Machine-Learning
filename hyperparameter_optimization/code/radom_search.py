# import libraries
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
model = LogisticRegression(solver = 'liblinear')

# create dictionary of hyperparameters
penalty_strength = [100, 10, 1.0, 0.1, 0.01]
penalty = ['l1', 'l2']
solvers = ['newton-cg', 'lbfgs', 'liblinear', 'saga']
hyperparameters = dict(C=penalty_strength, penalty=penalty, solver=solvers)

# grid search parameter tuning
randomizedsearch = RandomizedSearchCV(estimator=model, param_distributions=hyperparameters, n_iter=100, cv=3, random_state=7)
randomized_search.fit(x,y)

# print results
print('Best Score:', randomizedsearch.best_score_)
print('Best Solver:', randomizedsearch.best_estimator_.solver)
print('Best Penalty:', randomizedsearch.best_estimator_.penalty)
print('Best Penalty Strength:', randomizedsearch.best_estimator_.C)
