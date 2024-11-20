import sys 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC 
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt 
import numpy as np 

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
  # Let's apply rbf kernel of svm into our iris dataset
  iris = datasets.load_iris()
  X = iris.data[:, [2, 3]] # Petal Length and Petal Width 
  y = iris.target
  
  X_train, X_test, y_train, y_test = train_test_split(X, 
                                                      y, 
                                                      test_size=0.3, 
                                                      random_state=1)
  sc = StandardScaler()
  X_train_std = sc.fit_transform(X_train) 
  X_test_std = sc.transform(X_test)
  
  # Create subplots
  fig, ax = plt.subplots(1, 2, figsize=(12, 5))
  X_combined_std = np.vstack((X_train_std, X_test_std))
  y_combined = np.hstack((y_train, y_test))
  
  # SVM with gamma=0.1
  svm1 = SVC(kernel='rbf', random_state=1, gamma=0.1, C=1.0)
  svm1.fit(X_train_std, y_train)
  plt.sca(ax[0])  # Set the current axis
  plot_decision_regions(X_combined_std, 
                        y_combined, 
                        classifier=svm1, 
                        test_idx=range(105, 150))
  plt.title("SVM - gamma 0.1 - RBF kernel - Linearly Inseparable")
  plt.xlabel("Petal Width [standardized]")
  plt.ylabel("Petal Length [standardized]")
  plt.legend(loc="upper left")
  
  # SVM with gamma=100.0
  svm2 = SVC(kernel='rbf', random_state=1, gamma=100.0, C=1.0)
  svm2.fit(X_train_std, y_train)
  plt.sca(ax[1])  # Set the current axis
  plot_decision_regions(X_combined_std, 
                        y_combined, 
                        classifier=svm2, 
                        test_idx=range(105, 150))
  plt.title("SVM - gamma 100.0 - RBF kernel - Linearly Inseparable")
  plt.xlabel("Petal Width [standardized]")
  plt.ylabel("Petal Length [standardized]")
  plt.legend(loc="upper left")

# Apply tight layout and show the plot
plt.tight_layout()
plt.show()
