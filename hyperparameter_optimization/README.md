# An Introduction To Hyperparameter Optimization

## What Are Hyperparameters?

You can think of machine learning algorithms as systems with various knobs and dials, which you can adjust in any number of ways to change how output data (predictions) are generated from input data. The knobs and dials in these systems can be subdivided into parameters and hyperparameters. 

Parameters are model settings that are learned, adjusted, and optimized automatically. Conversely, hyperparameters need to be manually set manually by whoever is programming the machine learning algorithm. 

Generally, tuning hyperparameters has known effects on machine learning algorithms. However, it’s not always clear how to best set a hyperparameter to optimize model performance for a specific dataset. As a result, search strategies are often used to find optimal hyperparameter configurations. In this newsletter, I’m going to cover the following hyperparameter tuning methods: 
- **Grid Search** is a cross-validation technique for hyperparameter tuning that finds an optimal parameter value among a given set of parameters specified in a grid; and
- **Random Search** is a tuning technique that randomly samples a specified number of uniformly distributed algorithm parameters.

## Where Does Hyperparameter Tuning Fit In The Big Picture?

Before I show you how to use grid search and random search, you need to understand where these processes, or hyperparameter tuning more broadly, fit in the grand scheme of creating and deploying machine learning models. 

The image below visually represents the varying steps between identifying what type of machine learning problem you’re working with and deploying a functioning machine learning model that makes accurate predictions on unseen data. 

You can see that hyperparameter tuning is one of the final steps before you create a pipeline to automate your workflow and deploy your model, as demonstrated below:

<img src="images/DEA1.jpg" alt="Description" width="600" height="300">
