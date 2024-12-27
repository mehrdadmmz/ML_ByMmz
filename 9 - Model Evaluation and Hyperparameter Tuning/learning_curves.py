"""
Here, we will discuss how we can use learning curves to diagnose whether a learning algorithm
has a problem with overfitting (high variance) or underfitting (high bias).
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.model_selection import learning_curve, validation_curve

# Loading the Breast Cancer Wisconsin dataset 
df = pd.read_csv('https://archive.ics.uci.edu/ml/'
                 'machine-learning-databases'
                 '/breast-cancer-wisconsin/wdbc.data',
                 header=None)

X, y = df.loc[:, 2:].values, df.loc[:, 1]

# label encoding 
le = LabelEncoder()

# After encoding the class labels (diagnosis) in an array, y, the malignant tumors are now represented
# as class 1, and the benign tumors are represented as class 0, respectively
y = le.fit_transform(y) 

# Splitting data 
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    stratify=y, 
                                                    random_state=1)

# We can chain objects in a pipeline 
pipe_lr = make_pipeline(StandardScaler(), 
                        LogisticRegression(penalty='l2', 
                                           max_iter=10_000))

train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_lr, 
                                                        X=X_train, 
                                                        y=y_train, 
                                                        train_sizes=np.linspace(0.1, 1.0, 10), 
                                                        cv=10, 
                                                        n_jobs=1)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, 
         train_mean, 
         color='blue', 
         marker='o', 
         markersize=5, 
         label='Training accuracy')
plt.fill_between(train_sizes, 
                 train_mean - train_std, 
                 train_mean + train_std, 
                 alpha=0.15, 
                 color='blue')

plt.plot(train_sizes, 
         test_mean, 
         color='green', 
         linestyle='--',
         marker='s',
         markersize=5, 
         label='Validation accuracy')
plt.fill_between(train_sizes, 
                 test_mean - test_std, 
                 test_mean + test_std, 
                 alpha=0.15, 
                 color='green')

plt.grid()
plt.xlabel('Number of training examples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8, 1.03])
plt.show()

