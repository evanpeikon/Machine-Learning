# (1) import libraries
fron pandas import read_csv
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from math import sqrt

# (2) load data
fileame = 'insurance.csv'
names = ['age', 'sex', 'bmi', 'dependents', 'smoker', 'region', 'cost']
data = read_csv(filename, names=names)

# (3) create array w/ data and seperate array into input/outputs
array = data.values
x = array[:, 0:6] # inputs = all rows in columns 0-6
y = array[:, 6] # output = all rows in column 7

# (4) set algo (linear regression) and evaluation method (kfcv)
model = LinearRegression()
kfold = KFold(n_splits=10, random_state=5, shuffle=True)

# (5) set performance metric and evaluate model
scoring = 'neg_mean_squared_error'
results = cross_val_score(model, x, y, cv=kfold, scoring=scoring)
MSE = results.mean()*-1 # calculate mean MSE
RMSE = sqrt(MSE) # calcualte RMSE from MSE
print('RMSE:', RMSE)
