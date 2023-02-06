# Models Addition 

* **Support Vector Machines** - SVM (Support Vector Machine) is a type of supervised machine learning algorithm used for classification and regression tasks. The main 
idea behind SVM is to find the optimal boundaries (hyperplanes) that separate the data into different classes in a way that maximizes the margin or distance between the 
closest data points and hyperplanes in each class . These closest data points are called support vectors and form the support of the hyperplane. Hyperplanes are chosen 
to best generalize the data. This minimizes the risk of misclassifying new data points. For regression, the SVM algorithm attempts to fit a straight line that best 
predicts the target variable based on the input characteristics. SVMs are particularly effective when the data has a clear separation between classes, or when the number 
of features is much higher than the number of observations.

$$Gini(S) = 1 - ∑ (p_i)^2$$
$$Entropy(S) = - ∑ p_i log2(p_i)$$


* **Decsion Tree Classifier** - A decision tree classifier is a supervised machine learning algorithm used for classification problems. This is a tree-based model that 
recursively splits data into subsets based on feature values to make predictions. Each inner node of the tree represents a test of a feature and a branch represents a 
possible outcome of the test. A leaf of the tree represents the final predicted class of observations ending in that leaf. Finding the partition in the data that reduces 
class label contamination the most. This is usually measured by information gain or gini contamination. The algorithm can handle both categorical and numerical features 
and can be used for both binary and multiclass classification problems. Decision trees are easy to understand, interpret, and visualize, but they are prone to overfitting 
and instability when dealing with noisy or irrelevant features.  
