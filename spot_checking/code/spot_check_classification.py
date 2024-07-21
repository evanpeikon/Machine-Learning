# Import libraries
from pandas import read_csv
from sklearn.model_selection import kFold
from sklearn.model_selection import cross_val_score
from sklear.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.vsm import SCV
from matplotlib import pyplot

# load data
filename = 'diabetes.csv'
names = ['pregnancies', 'glucose', 'DBP', 'skinfold','insulin','BMI','diabetes_pedigree','age','outcome']
data = read_csv(filename, names=names)

# create array w/ data and seperate array into input/outputs
array = data.values # inputs = all rows in columns 0-7
x = array[:,0:8] # output = all rows in column 8
y= array[:,8]

# prepare models for spot-check and comparison
models= []
model.append(('LR', LogisticRegression(solver='liblinear')))
model.append(('LDA', LinearDiscriminantAnalysis()))
model.append(('KNN', KNeighborsClassifier()))
model.append(('CART', DecisionTreeClassifier()))
model.append(('NB', GaussianNB()))
model.append(('SVM', SVC()))

# evaluate model performance and summarize results
results = []
names = []
scoring = 'accuracy'
for name, model in models:
  kfold = kFold(n_splits=10, random_state=5, shuffle=True)
  cv_results = cross_val_score(model, x, y, cv=kfold, scoring=scoring)
  results.append(cv_results)
  names.append(name)
  read_out = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
  print(read_out)

# create boxplot to compare results
fig = pyplot.figure()
fig.subtitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()
