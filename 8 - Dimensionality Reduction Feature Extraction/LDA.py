import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

# computing mean vectors
# 
# Set printing options
# These options determine the way floating point numbers, arrays and other NumPy objects are displayed.
# shape of each mean vector will be one dim vector of 13 elements, each of which is the mean of each  
# feature that belongs to class i in {1, 2, 3} here. 
# Like for class 1, we filter everything that is for class 1 first, then we will find the mean of each feature
# its shape will be (13, )
np.set_printoptions(precision=4)
mean_vecs = []
for label in range(1, 4): 
    mean_vecs.append(np.mean(X_train_std[y_train == label], axis=0))
    print(f"MV {label}: {mean_vecs[label - 1]}\n")
    
d = 13 # number of features
mean_overall = np.mean(X_train_std, axis=0).reshape(d, 1)

# Within-class scatter matrix S_W
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.cov(X_train_std[y_train == label].T)
    S_W += class_scatter
    
print('Within-class scatter matrix: '
      f'{S_W.shape[0]}x{S_W.shape[1]}')

# Between-class scatter matrix S_B
S_B = np.zeros((d, d))
for i, mean_vec in enumerate(mean_vecs):
    n = X_train_std[y_train == i + 1, :].shape[0]
    mean_vec = mean_vec.reshape(d, 1) # make column vector
    S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)

print('Between-class scatter matrix: '
      f'{S_B.shape[0]}x{S_B.shape[1]}')

# eigenvlaues, eigenvectors, and eigenpairs of the (SW)^-1(SB)
eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]

# Sorting them (decreasing from largest to the smallest) based on the eigen val and its associated eigen vec
eigen_pairs = sorted(eigen_pairs, key=lambda k : k[0], reverse=True)
print("\nEigenvalues in descending order: \n")

# In LDA, the number of linear discriminants is at most c-1 with c being the number of classes
for eigen_val in eigen_pairs: 
    print(eigen_val[0])

# measure how much of the class-discriminatory information is captured by the linear discriminants
# (eigenvectors), let’s plot the linear discriminants by decreasing eigenvalues
tot = sum(eigen_vals.real)
discr = [(i / tot) for i in sorted(eigen_vals.real, reverse=True)]
cum_discr = np.cumsum(discr)

# We can see that the first two linear discriminants alone capture 100 percent of the useful
# information in the Wine training dataset
plt.bar(range(1, 14), discr, align="center", label="Individual discriminability")
plt.step(range(1, 14), cum_discr, where='mid', label="Cumulative discriminability")
plt.ylabel('"Discriminability" ratio')
plt.xlabel("Linear Discriminants")
plt.ylim([-0.1, 1.1])
plt.legend(loc="best")
plt.tight_layout()
plt.show()

# Stacking the two most discriminative eigenvector coulumns to create the transformation matrix W: 
W = np.hstack((eigen_pairs[0][1][:, np.newaxis].real, 
               eigen_pairs[1][1][:, np.newaxis].real))
print("Matrix W: \n", W)

# Projecting examples onto the new feature space: X' = XW
X_train_lda = X_train_std.dot(W)
colors = ['r', 'b', 'g']
markers = ['o', 's', '^']
for l, c, m in zip(np.unique(y_train), colors, markers): 
    plt.scatter(X_train_lda[y_train == l, 0], 
                X_train_lda[y_train == l, 1] * (-1), 
                c=c, label=f'Class {l}', marker=m)

plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
