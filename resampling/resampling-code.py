# Import libraries
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# Load data
filename = 'diabetes.csv'
names = ['pregnancies', 'glucose', 'DBP', 'skinfold','insulin','BMI','diabetes_pedigree','age','outcome']
data = read_csv(filename, names=names)

# create array w/ data and seperate array into input/outputs
array = data.values # inputs = all rows in columns 0-7
x = array[:,0:8] # output = all rows in column 8
y= array[:,8]

# create train-test-split
test_split = 0.35 # sets test split's size to 35% of dataset
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = test_split, random_state = 5)

# train test split
model = LogisticRegression(solver = 'liblinear')
model.fit(x_train, y_train)
result = model.score(x_test, y_test)
print('Accuracy is...', result*100)

# k-fold cross-validation
k_fold = kFold(n_splits = 5, random_state = 5, shuffle=True)
model = LogisticRegression(solver = 'liblinear')
results = cross_val_score(model, x, y, cv=k_fold)
print('Results per split:', results)
print('Mean result:', results.mean()*100)
print('Std.Dev of results:', results.std()*100)

# leave one out cross-validation
loocv = LeaveOneOut()
model = LogisticRegression(solver = 'liblinear')
results = cross_val_score(model, x, y, cv=loocv)
print('Mean result (accuracy):, results.mean()*100)
print('Std.Dev of results:', results.std()*100)
