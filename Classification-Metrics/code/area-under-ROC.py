# Import libraries
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection mport cross_val_score
from sklearn.linear_model import LogisticRegression

# Load data
filename = 'diabetes.csv'
names = ['pregnancies', 'glucose', 'DBP', 'skinfold','insulin','BMI','diabetes_pedigree','age','outcome']
data = read_csv(filename, names=names)

# create array w/ data and seperate array into input/outputs
array = data.values # inputs = all rows in columns 0-7
x = array[:,0:8] # output = all rows in column 8
y= array[:,8]

# Set algo (logistic regression) and evaluation method (k-fold cv)
kfold = KFold(n_splits=10, random_state=5, shuffle=True)
model = LogisticRegression(solver= 'liblinear')

# Set performance metric (Area under ROC curve) and evaluate model
score = 'roc_curve'
results = cross_val_score(model,x,y,cv=kfold,scoring=score)
print('Area under ROC curve mean:', results.mean())
print('Area under ROC curve SD:', results.std())
