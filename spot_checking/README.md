# Comparing Machine Learning Models
For any machine learning problem, we must select an algorithm to make predictions, an evaluation method to estimate a model's performance on unseen data, and an evaluation metric(s) to quantify how well the model works. 

Unfortunately, we can't always know which algorithm will work best on our dataset beforehand. As a result, we have to try several algorithms, then focus our attention on those that seem most promising. Thus, it's important to have quick and easy ways to assess and compare different algorithms' performance before we select one to tune and optimize - this is where spot-checking comes in. 

Spot-checking is a way to quickly discover which algorithms perform well on your machine-learning problem before selecting one to commit to. Generally, I recommend that you spot-check five to ten different algorithms using the same evaluation method and evaluation metric to compare the model's performance. Often you'll notice that a few of your spot-checked algorithms perform much better than the rest. I recommend selecting two to three models with the best performance, then double down and spend time tuning those algorithms to make your predictions increasingly accurate. 

# Algorithm Spot-Checking For Classification Problems:

A classification problem in machine learning is one in which a class label is predicted given specific examples of input data. 

In this article, I will use the Pima Diabetes dataset for my demonstration. The Pima Diabetes dataset is a binary classification problem with two numerical output classes (0 = no diabetes and 1 = diabetes). 

The key to fairly comparing different spot-checked machine learning algorithms is to evaluate each algorithm in the same way, which is achieved with a standardized test harness. In this article, each algorithm we spot-check will be used to predict whether a given patient has diabetes, using k-fold cross-validation and classification accuracy as our evaluation method and evaluation metric, respectively.

Finally, we will spot-check the following six algorithms:

- **Logistic regression**, which is a data analysis technique that uses several known input values to predict a single unknown data point; 
- **Linear discriminant analysis (LDA)((, which makes predictions for both binary and multi-class classification problems; 
- **k-nearest neighbors (KNN)**, which finds the k most similar training data points for a new instance and takes the mean of the selected training data points to make a prediction;
- **Naive bayes (NB)**, which calculates the probability and conditional probability of each class, given each input value, and then estimates these probabilities for new data;
- **Classification and regression trees (CART)**, which construct binary tress from the training data and generate splits to minimize a cost function; and
- **Support vector machine (SVM)**, which seek a line that best separates to classes based on the position of various support vectors.

Below you’ll find sample code spot-checking the six algorithms above and generating a box plot to compare the results:

```
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
```
Which produces the following output:
