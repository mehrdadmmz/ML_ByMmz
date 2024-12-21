"""
LDA (linear discriminant analysis) like PCA is a linear transformation technique that can reduce 
the number of dimensions in a dataset, Unlike PCA which was unsupervised, LDA is a supervised method. 
SO, it takes class label information into account. 

Inner working of LDF: 
1. Standardize the d-dimensional dataset (d is the number of features).
2. For each class, compute the d-dimensional mean vector.
3. Construct the between-class scatter matrix, SB, and the within-class scatter matrix, SW.
4. Compute the eigenvectors and corresponding eigenvalues of the matrix, (SW)^-1(SB).
5. Sort the eigenvalues by decreasing order to rank the corresponding eigenvectors.
6. Choose the k eigenvectors that correspond to the k largest eigenvalues to construct a d×k-dimensional
transformation matrix, W; the eigenvectors are the columns of this matrix.
7. Project the examples onto the new feature subspace using the transformation matrix, W.
"""

