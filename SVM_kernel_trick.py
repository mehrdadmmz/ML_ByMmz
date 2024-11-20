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

if __name__ == "__mian__": 
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
  X_test = sc.transform(X_test)
  
  # Create subplots
  fig, ax = plt.subplots(1, 2, figsize=(12, 5))
  
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
