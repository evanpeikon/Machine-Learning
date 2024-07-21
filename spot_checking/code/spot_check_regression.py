# import libraries
from pandas import read_csv
from sklearn.model_selection import kFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

# load data
fileame = 'insurance.csv'
names = ['age', 'sex', 'bmi', 'dependents', 'smoker', 'region', 'cost']
data = read_csv(filename, names=names)

# create array w/ data and seperate array into input/outputs
array = data.values
x = array[:, 0:6] # inputs = all rows in columns 0-6
y = array[:, 6] # output = all rows in column 7

# prepare models for spot-check and comparison
models = []
models.append(('LR:', LinearRegression()))
models.append(('Ridge:', Ridge()))
models.append(('Lasso:', Lasso()))
models.append(('ENR:', ElasticNet()))
models.append(('KNNR:', kNeighborsRegressor()))
models.append(('CART:', DecisionTreeRegressor()))
models.append(('SVM:', SVR()))

# evaluate model performance and summarize results
results = []
names = []
scoring = 'neg_mean_squared_error'
for name, model in models:
  kfold = kFold(n_splits=10, random_state=5, shuffle=True)
  cv_results = cross_val_score(model, x, y, cv=kfold, scoring=scoring)
  results.append(cv_results)
  names.append(name)
  read_out = "%s: %f (%f)" (name, cv_results.mean(), cv_results.std())
  print(read_out)

# create boxplot to compare results
fig = pyplot.figure()
fix.subtitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()
