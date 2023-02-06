## PCA(Principal Component Analysis)
PCA (Principal Component Analysis) is a dimensionality reduction technique commonly used in machine learning. It is a method of transforming the data into a new 
coordinate system such that the first new axis (principal component) has the highest variance, the second new axis has the second highest variance and so on. This way, 
it reduces the number of features in the data while preserving the most important information. The main use of PCA is for visualization purposes, to detect patterns in 
the data and to remove noise and redundant features, making the data easier to work with and potentially improving the performance of machine learning algorithms.
The mathematics behind PCA can be broken down into the following steps:
* **Center data**: The first step is to center the data by subtracting the mean from each feature. Centered data are represented as a matrix X, where each row represents 
an observation and each column represents a feature. 

* **Covariance matrix: **
$$C = \frac {1}{n-1} X^TX$$

* **Eigenvectors and Eigenvalues**: Then the eigenvectors and eigenvalues of the covariance matrix are computed. The eigenvectors represent the principal components of 
the data, and the eigenvalues represent the amount of variance explained by each principal component.

$$Ax = \lambda x$$

* **Project of vectors**: The final step is to project the centered data onto the principal components by taking the inner product of the centered data matrix and the 
matrix of eigenvectors. The resulting matrix represents the transformed data in the new coordinate system defined by the principal components. 

$$proj_va = \frac {a.v}{||v||^2}v$$
