import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

"""
Sometimes, we are interested to know about how much each original
feature contributes to a given principal component. These contributions are often called loadings.
The factor loadings can be computed by scaling the eigenvectors by the square root of the eigenvalues.
The resulting values can then be interpreted as the correlation between the original features and
the principal component.

# In sklearn: 
# eigenvectors -- > pca.components_ 
# eigenvalues  -- > pca.explained_variance_

After plotting the bar we can see that: 
for example, Alcohol has a negative correlation with the first principal
component (approximately –0.3), whereas Malic acid has a positive correlation (approximately 0.54).
Note that a value of 1 describes a perfect positive correlation whereas a value of –1 corresponds to a
perfect negative correlation
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

# initializing the PCA transformer
pca = PCA(n_components=2)

# dimensionality reduction:
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

# Loadings
sklearn_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

# Bar plot for both PC 1 and PC 2 comparing to the loadings
n_features = X_train_std.shape[1]

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

plt.sca(axes[0])
plt.bar(range(n_features), sklearn_loadings[:, 0], align="center")
plt.title("Loadings for PC 1")
plt.xticks(range(n_features))
plt.xticklabels(df_wine.columns[1:], rotation=90)
plt.ylabel("Loadings")
plt.ylim([-1, 1])

plt.sca(axes[1])
plt.bar(range(n_features), sklearn_loadings[:, 1], align="center")
plt.title("Loadings for PC 2")
plt.xticks(range(n_features))
plt.xticklabels(df_wine.columns[1:], rotation=90)
plt.ylim([-1, 1])

plt.tight_layout()
plt.show()

