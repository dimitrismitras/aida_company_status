# company_status
Classification python code, for companies status 

### Check report for details and results

GitHub Repository: https://github.com/dimitrismitras/aida_company_status

### Authors: 
Dimitrios Mitras aid24005, Konstantinos Bougioukas aid24011

### Theme: 
Comparison of classification methods for predicting whether a company will go bankrupt or not based on the performance and activity indicators of the company.

### Introduction
The two basic categories of machine learning are supervised and unsupervised machine
learning. In supervised learning, the model is trained on a labelled dataset, meaning
that the input data is paired with corresponding output values/labels. Supervised
learning problems are applicable to a variety of situations and data types. Common
tasks in supervised learning include regression modelling for continuous output and
classification when the output is discrete values or classes.
In this homework, our focus is on the case of binary classification modelling, where the
output is a binary variable. Specifically, our aim is to compare and evaluate the
performance of different classification methods in both unbalanced and balanced
training datasets.

### Problem description
We want to train a model that best predicts whether a company is healthy or will go
bankrupt based on performance and activity indicators of the company. This is a binary
classification problem.

### Dataset description
The performance of the classification algorithms will be studied in a dataset named
“Dataset2Use_Assignment1” that contains 10716 observations and 13 columns. We are
interested in the performance indicators of the companies (columns from A to H) and
three binary indicators of activities (columns from I to K) which are referred to as
features, input variables, or predictors of the dataset. The target variable for the models
is the status of the company (healthy is coded with 1, and bankrupt is coded with 2).

### Classification methods used in this work
a. Linear Discriminant Analysis (LDA): The LDA provides a probabilistic framework
for classification and is particularly useful in scenarios where we have a dataset with
multiple classes and we want to find a linear combination of features that best separates
(discriminates) these classes (Rao 1973). The main goal of LDA is to maximise the
distance between the means of different classes while minimising the spread or
variance within each class. LDA assumes that the features are continuous and normally
distributed, and that classes have the same covariance matrix. Note that we have a mix
of continuous and binary variables as input. However, LDA is quite robust to the
violation of the assumptions.

b. Logistic Regression: Logistic Regression is specifically designed for binary
classification. It uses the logistic function (sigmoid function) to map the linear
Machine Learning and Computer Vision Homework 1: Supervised Machine Learning (Classification)
combination of input features to a range between 0 and 1 (ref). Logistic regression is a
direct probability model and doesn't require Bayes' rule, unlike discriminant analysis,
for the conversion of outcomes into probabilities.

c. Decision Trees: A decision tree is a flowchart-like tree structure that operates by
recursively partitioning the input data into subsets based on the values of different
features. At each internal node of the tree, a decision is made regarding which feature
to split on, and these decisions lead to the creation of branches that ultimately lead to
the assignment of a class label at the leaf nodes.

d. Random Forests: A Random Forest combines the output of multiple decision trees
(ensemble of trees) to reach a single result, usually trained with the bagging method.
The general idea of the bagging method is that a combination of learning models
increases the overall result.

e. K-Nearest Neighbours: The K-Nearest Neighbor (KNN) algorithm is a
non-parametric supervised machine learning model. This is a very simple concept of
identifying K-nearest neighbours for a given data point. The assumption is that similar
items are closer together. Here the idea of close is a distance measure between two
points. By finding the largest class of items that are close to the test data, we can
conclude that test data belongs to that class.

f. Naïve Bayes: It is a probabilistic machine learning algorithm based on Bayes'
theorem. It is considered "naïve" because it makes the assumptions that the features
used to describe an observation are normally distributed and conditionally
independent, given the class label. This method can be extremely fast relative to other
classification algorithms.

g. Support Vector Machines: In machine learning, SVM is used to classify data by
finding the optimal decision boundary, a hyperplane, that maximally separates
different classes. This method offers valuable theoretical understanding of the two-class
classification process, especially when assuming the linear separability of the data. If
the class boundaries are not linearly separable, SVM uses a technique where the
dimensional representation can be converted to higher dimension data. In this higher
dimensional representation, the class boundaries become linearly separable, and SVM
classifiers can provide a boundary.

h. Gradient Boosting Classifier: The idea behind Gradient Boosting is to build trees
sequentially, with each tree correcting the errors of the previous ones. The process
involves minimising a loss function (often a negative log-likelihood), and the learning is
done by optimising the parameters of each weak learner (tree).


### Performance Metrics
In order to objectively evaluate our results, five different metrics are considered based
on the confusion matrix (Table 1): Accuracy, Precision, Recall, the F1 score, and the
AUC ROC.

Accuracy (ACC) is defined as: ACC = (TP + TN) / (TP + FP + TN + FN). Percentage of
correct classification for all classes.

Precision (Pr) is defined as: Pr = TP / (TP + FP). It calculates how many correct positive
predictions we have to the total predicted positive observations (also known as Positive
Predictive Value - PPV).

Recall (Re) is defined as: Re = TP / (TP + FN). It represents the ratio of correctly
predicted positive observations to the total actual positive observations. (also known as
Sensitivity and True Positive Rate)

F1 score is defined as: F1 = (Pr × Re)/(Pr + Re). It calculates the weighted harmonic mean
of precision and recall.

Specificity (True Negative Rate) is defined as: TNR = TN / (TN + FP). It measures the
proportion of actual negative instances that are correctly identified as negative by the
classification model.

