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

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    stratify=y, 
                                                    random_state=1)

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)

# Training a SVM model 
svm = SVC(kernel="linear", 
          C=1.0, 
          random_state=1)
svm.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, 
                      y_combined, 
                      classifier=svm, 
                      test_idx=range(105, 150))

plt.title("SVM classifier on Iris data")
plt.xlabel("Petal Length [standardized]")
plt.ylabel("Petal Width [standardized]")
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()
