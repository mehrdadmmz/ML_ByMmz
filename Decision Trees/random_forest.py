import numpy as np 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap

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
  iris = datasets.load_iris()
  X = iris.data[:, [2, 3]] # Petal Length and Petal Width 
  y = iris.target
  
  X_train, X_test, y_train, y_test = train_test_split(X, 
                                                      y, 
                                                      test_size=0.3, 
                                                      random_state=1, 
                                                      stratify=y)

  # n_estimator = the # of trees in the forest.
  # n_jobs = allows us to parallelize the model training using multiple cores of our computer
  # criterion is gini by default
  # sklearn chooses all the sample size by default
  forest = RandomForestClassifier(n_estimators=25, 
                                  random_state=1, 
                                  n_jobs=2,)
  forest.fit(X_train, y_train)
  
  X_combined = np.vstack((X_train, X_test))
  y_combined = np.hstack((y_train, y_test))
  plot_decision_regions(X_combined, 
                        y_combined, 
                        classifier=forest, 
                        test_idx=range(105, 150))
  plt.xlabel("Petal Length [cm]")
  plt.ylabel("Petal Width [cm]")
  plt.legend(loc="upper left")
  plt.tight_layout()
  plt.show()
  
  print(f"Training accuracy: {forest.score(X_train, y_train):.2f}")
  print(f"Testing accuracy: {forest.score(X_test, y_test):.2f}")
