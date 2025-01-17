import sys 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
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
        
# 1 
iris = datasets.load_iris()

X = iris.data[:, [2, 3]]
y = iris.target

print("Class labels: ", np.unique(y))

# stratify: If not None, data is split in a stratified fashion, using this as the class labels
#stratify makes sure that both training and test dataset have the same class proportion as the original dataset 
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                  test_size=0.3, 
                                                  random_state=1, 
                                                  stratify=y)

print("Lebels counts in y        ", np.bincount(y))
print("Lebels counts in y_train  ", np.bincount(y_train))
print("Lebels counts in y_test   ", np.bincount(y_test))

# feature scaling 
sc = StandardScaler()
# using fit method, we will estimate the mean and std of each feature dimension from the training data
sc.fit(X_train)
X_train_std = sc.transform(X_train) 
X_test_std = sc.transform(X_test)
# after standardizing the training data, now we can train a peceptron model 
ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)
y_pred = ppn.predict(X_test_std)
print(f"Missclassified examples: {(y_test != y_pred).sum()}")

# measuting the accuracy using the metric class
print(f"Accuracy {accuracy_score(y_test, y_pred):,.3}")
# combining the predict and accuracy_score and output the accuracy of a model 
print(f"Accuracy {ppn.score(X_test_std, y_test):,.3}")

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined_std, 
                    y=y_combined, 
                    classifier=ppn, 
                    test_idx=range(105, 150))
plt.xlabel("Petal length [standardized]")
plt.ylabel("Petal width [standardized]")
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()


# 2
def sigmoid(z): 
    return 1.0 / (1.0 + np.exp(-z))

# let's plot the sigmoid func for values in the range -7 to 7 
z = np.arange(-7, 7, 0.1)
sigma_z = sigmoid(z)

plt.plot(z, sigma_z)
# Add a vertical line across the Axes.
plt.axvline(0.0, color="k")
plt.ylim(-0.1, 1.1)
plt.xlabel("z")
plt.ylabel("$\sigma (z)$")

# y axis ticks and gridline 
plt.yticks([0.0, 0.5, 1.0])
# Get the current Axes. If there is currently no Axes on this Figure
ax = plt.gca()
ax.yaxis.grid(True)
plt.tight_layout()
plt.show()

 

