# Import libraris
from pandas import read_csv
from sklearn.model_selection import tran_test_split
from sklearn.model_selection mport LogisticRegression
from sklearn.linear_model import confusion_matrix

# Load data
filename = 'diabetes.csv'
names = ['pregnancies', 'glucose', 'DBP', 'skinfold','insulin','BMI','diabetes_pedigree','age','outcome']
data = read_csv(filename, names=names)

# Create array w/ data and seperate array into input/outputs
array = data.values # inputs = all rows in columns 0-7
x = array[:,0:8] # output = all rows in column 8
y= array[:,8]

# Create train-test split
test_split = 0.35 # Sets the test split's size to 35% of dataset
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=test_split, random_state=seed)

# Craeate and fit model
model = LogisticRegression(solver= 'liblinear')
model.fit(x_train,y_train)

# Make predictions and generate confusion matrix
predictions = model.predict(x_test)
confusion_matrix = confusion_matrix(y_test, predictions)
print(confusion_matrix)
