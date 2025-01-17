import sys 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Perceptron, LogisticRegression, SGDClassifier
from sklearn.svm import SVC 
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

if __name__ == "__main__": 
    # Let's create a linearly inseprable data 100 with class label of 1 and 100 with class label of 0

    np.random.seed(1)

    X_xor = np.random.randn(200, 2)
    y_xor = np.logical_xor(X_xor[:, 0] > 0, 
                           X_xor[:, 1] > 0)
    y_xor = np.where(y_xor, 1, 0)

    # Create subplots
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns

    # Plot the first scatter plot
    ax[0].scatter(X_xor[y_xor == 1, 0], 
                  X_xor[y_xor == 1, 1], 
                  c="royalblue", 
                  marker="s", 
                  label="Class1")
    ax[0].scatter(X_xor[y_xor == 0, 0], 
                  X_xor[y_xor == 0, 1], 
                  c="tomato", 
                  marker="o", 
                  label="Class0")
    ax[0].set_xlim([-3, 3])
    ax[0].set_ylim([-3, 3])
    ax[0].set_xlabel("Feature 1")
    ax[0].set_ylabel("Feature 2")
    ax[0].legend(loc="best")
    ax[0].set_title("Scatter Plot")

    # Train SVM
    svm = SVC(kernel="rbf", random_state=1, gamma=0.10, C=10.0)
    svm.fit(X_xor, y_xor)

    # Plot decision regions
    plt.sca(ax[1])  # Set the second subplot as the current axis
    plot_decision_regions(X_xor, y_xor, classifier=svm)
    ax[1].legend(loc="best")
    ax[1].set_title("Decision Regions")

    # Adjust layout and show plots
    plt.tight_layout()
    plt.show()
