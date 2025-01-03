import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

"""PCA 

1. Standardize the d-dimensional dataset.
2. Construct the covariance matrix.
3. Decompose the covariance matrix into its eigenvectors and eigenvalues.
4. Sort the eigenvalues by decreasing order to rank the corresponding eigenvectors.
5. Select k eigenvectors, which correspond to the k largest eigenvalues, where k is the dimensionality
of the new feature subspace (𝑘𝑘 𝑘 𝑘𝑘 ).
6. Construct a projection matrix, W, from the “top” k eigenvectors.
7. Transform the d-dimensional input dataset, X, using the projection matrix, W, to obtain the
new k-dimensional feature subspace.

xW = z which x is a d dim, W is a d*k dim, and z will be a k dim (k << d)"""

# Load the dataset
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                   'machine-learning-databases/wine/wine.data', 
                   header=None)

# Setting up the columns
df_wine.columns = ['Class label', 'Alcohol',
                'Malic acid', 'Ash',
                'Alcalinity of ash', 'Magnesium',
                'Total phenols', 'Flavanoids',
                'Nonflavanoid phenols',
                'Proanthocyanins',
                'Color intensity', 'Hue',
                'OD280/OD315 of diluted wines',
                'Proline']

# split the data
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    random_state=0, 
                                                    stratify=y)

# Standardize the dataset
stdscl = StandardScaler()
X_train_std = stdscl.fit_transform(X_train) 
X_test_std = stdscl.transform(X_test)

# Covariance matrix 13 * 13 
cov_mat = np.cov(X_train_std.T)

# Eigen values and eigen vectors 
"""A related function, numpy.linalg.eigh, has been implemented to decompose Hermetian
matrices, which is a numerically more stable approach to working with symmetric matrices
such as the covariance matrix; numpy.linalg.eigh always returns real eigenvalues."""
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

# eigen_vals is a row vector of 13 elements which represent the eigen values
# corresponding eigen vectors is stored as columns in a 13 * 13 dim matrix in eigen_vecs
print("\nEigenvalues \n", eigen_vals)

# The variance explained ratio of an eigenvalue, 𝜆𝑗, is simply the
# fraction of an eigenvalue, 𝜆𝑗, and the total sum of the eigenvalues
# Explained variance ratio = lambda j / sum of lambdas j such that j = 1 to j = dtot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]

# cumulative sum of explained variances
cum_var_exp = np.cumsum(var_exp)

# plotting (indiv exp var)
plt.bar(range(1, 14), var_exp, align="center", label="Individual explained variance")

# plotting (cumulative exp var)
plt.step(range(1, 14), cum_var_exp, where="mid", label="Cumulative explained variance")

plt.ylabel("Explained variance ratio")
plt.xlabel("Principal component index")
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low (descending)
eigen_pairs.sort(key=lambda k: k[0], reverse=True)

# Building the W matrix (projection matrix d * k dim) by grabbing the k biggest eigenvectors associated with the k biggest eigenvalues
w = np.hstack((eigen_pairs[0][1][:, np.newaxis], 
               eigen_pairs[1][1][:, np.newaxis]))

# 13 by 2 matrix so now we can convert dataset from 13 features down to 2
print("Matrix W:\n", w)

# transforming the entire dataset 
# X' = XW (X' is now in the shape of 124 * 2 which means has only 2 features)
# X' is now transformed into PCA subspace (the principal components one and two)
X_train_pca = X_train_std.dot(W)
print("The transformed dataset: ", X_train_pca)

# plot the transformed Wine training dataset, now stored as 124 * 2-dimension
colors = ['r', 'b', 'g']
markers = ['o', 's', '^']

for l, c, m in zip(np.unique(y_train), colors, markers): 
    plt.scatter(X_train_pca[y_train == l, 0],  # Filter for the current class label `l`
                X_train_pca[y_train == l, 1],
                c=c, label=f"Class {l}", marker=m)

plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.legend(loc="lower left")
plt.tight_layout()
plt.show()
