# What Are Evaluation (Performance) Metrics?

In a previous tutorial titled [Evaluating Machine Learning Models: An Introduction to Resampling Methods](https://github.com/evanpeikon/machine-learning/tree/main/resampling), I discussed evaluation methods, which estimate how well a given machine learning algorithm will perform at making predictions about unseen data. Example evaluation methods include train-test splitting, k-fold cross-validation, and leave one out cross-validation.

Whereas evaluation methods estimate a model's performance on unseen data, evaluation metrics are the statistical techniques employed to quantify how well the model works. Thus, to evaluate a machine learning algorithm, we must select an evaluation method and evaluation metric(s). 

In the aforementioned tutorial I covered a handful of common evaluation methods, using accuracy as the chosen evaluation metric in each case. In this tutorial, I will focus on classification problems, using the Pima Diabetes dataset for my demonstrations. The Pima Diabetes dataset is a binary classification problem with numerical output classes (0 = no diabetes and 1 = diabetes). Additionally, in all of the code examples, I will use the same algorithm (logistic regression) and evaluation method (k-fold cross-validation) while demonstrating the following evaluation metrics:
- **Classification accuracy**, which is the percent of all predictions made correctly;
- A **confusion matrix**, which is a table summarizing the prediction results for a classification problem;
- A **classification report**, which provides a convenient snapshot of a machine learning model’s performance on classification problems;
- The **area under the ROC curve**, which represents a machine learning model’s ability to accurately discriminate between output classes; and
- **Logistic loss**, which represents the confidence for a given algorithms predictive capabilities.

# Classification Accuracy:

Classification accuracy is the percentage of all predictions that are made correctly. 

Because it's so easy to calculate and interpret, classification accuracy is the most commonly used evaluation metric for classification problems. However, classification accuracy is only effective when there are equal numbers of observations in each output class, which is seldom the case, and thus classification accuracy is often misused. For example, let's say we have 100 patients, 10 of whom are diseased. Our machine learning algorithm predicted that 95 patients were healthy and 5 were diseases. In this case, our classification accuracy would be 90%, which while true, is misleading given that we misclassified 50% of diseased patients (a catastrophic outcome in real life). 

Below you'll find sample code using classification accuracy (in conjunction with k-fold cross-validation) to determine the effectiveness of a machine learning model:
```
# Import libraris
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection mport cross_val_score
from sklearn.linear_model import LogisticRegression

# Load data
filename = 'diabetes.csv'
names = ['pregnancies', 'glucose', 'DBP', 'skinfold','insulin','BMI',diabetes_pedigree','age','outcome']
data = read_csv(filename, names=names)

# create array w/ data and seperate array into input/outputs
array = data.values # inputs = all rows in columns 0-7
x = array[:,0:8] # output = all rows in column 8
y= array[:,8]

# Set algo (logistic regression) and evaluation method (k-fold cv)
kfold = KFold(n_splits=10, random_state=5, shuffle=True)
model = LogisticRegression(solver= 'liblinear')

# Set performance metric and evaluate model
score = 'accuracy'
results = cross_val_score(model, x, y, cv=kfold, scoring=score)
print('Mean Accuracy:', results.mean()*100)
print('Accuracy Std.Dev.:', results.std())
```
Which results in the following outputs:
- Mean Accuracy: 77.08%
- Accuracy Std.Dev.: 0.05

# Confusion Matrix:
A confusion matrix is a table that summarizes the prediction results for a classification problem with the predicted values on the x-axis and actual values on the y-axis, as demonstrated below:

