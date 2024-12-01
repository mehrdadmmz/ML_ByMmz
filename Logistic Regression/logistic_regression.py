from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Perceptron, LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=f'Class {cl}', 
                    edgecolor='black')

    # highlight test examples
    if test_idx:
        # plot all examples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='none',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100, 
                    label='Test set')        

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    stratify=y, 
                                                    random_state=1)

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# solving optimization problem using: 
# solver = "limited Broyden, Fletcher, Goldfarb and Shanno" algorithms which is great for multiclass 
# Parameter C controls the strength of regularization and is inversly proportioal to the lambda whichi 
# is the reg parameter. higher C, lower reg. lower C, higher reg. 
lr  = LogisticRegression(C=100.0, 
                         solver="lbfgs", 
                         multi_class="ovr") # can also be: multi_class="multinomial" 

# training our model using .fit()
lr.fit(X_train_std, y_train)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined_std, 
                      y_combined, 
                      classifier=lr, 
                      test_idx=range(105, 150))

plt.title("Logistic regression on Iris data")
plt.xlabel("Petal length [standardized]")
plt.ylabel("Petal width [standardized]")
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()

# The .coef_ attribute in a scikit-learn linear model (such as Logistic Regression, Linear Regression, 
# or similar models) represents the coefficients (weights) of the linear decision boundary (or hyperplane)
# in the feature space.

# Each row corresponds to one class in a one-vs-rest (OvR) scheme.
# Each column corresponds to the weight (coefficient) assigned to a 
# particular feature for separating the corresponding class from the others.
print(lr.coef_)

# the prob that training examples belong to a certain class can be computed using 
# .predict_proba method. first row is for the first flower and each col of it is
# the prob that flower belong to class i (here we have 3 classes for flowers 0, 1, 2)
print(lr.predict_proba(X_test_std[:3, :]))

# better way of obtaining the class label is by using the predict method 
lr.predict(X_test_std[:3, :])
