# Grid search cross-validation
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

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

# Pipeline setup:
# StandardScaler performs feature scaling to standardize the dataset 
# (mean = 0, standard deviation = 1), which is essential for SVM performance.
# SVC (Support Vector Classifier) is the classifier model used with a specified random state for reproducibility.
pipe_svc = make_pipeline(StandardScaler(), 
                         SVC(random_state=1))  

# Define the range of hyperparameters to tune for the SVC model
# 'C' is the regularization parameter that controls the trade-off between maximizing the margin 
# and minimizing classification error.
# 'gamma' defines the influence of a single training example in RBF kernels.
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

# Create the parameter grid for GridSearchCV:
# 1. Linear kernel: Only tune the 'C' parameter.
# 2. RBF (Radial Basis Function) kernel: Tune both 'C' and 'gamma'.
param_grid = [
    {'svc__C': param_range,                # Linear kernel grid
     'svc__kernel': ['linear']},           # Kernel type is linear
    {'svc__C': param_range,                # RBF kernel grid
     'svc__gamma': param_range,            # Tuning both 'C' and 'gamma'
     'svc__kernel': ['rbf']}               # Kernel type is RBF
]

# GridSearchCV setup:
# Perform exhaustive search over the parameter grid using cross-validation.
# - estimator: The pipeline (scaler + SVC model).
# - param_grid: The dictionary of hyperparameters to search.
# - scoring: Use accuracy as the evaluation metric.
# - cv: 10-fold cross-validation.
# - refit: Automatically refit the best model on the entire training dataset after finding the best parameters.
# - n_jobs: Set to -1 to use aall CPU cores for faster process.
gs = GridSearchCV(estimator=pipe_svc, 
                  param_grid=param_grid, 
                  scoring='accuracy', 
                  cv=10, 
                  refit=True, 
                  n_jobs=-1)

# Fit the grid search to the training data to find the best hyperparameters
gs = gs.fit(X_train, y_train)

# Output the best cross-validation accuracy score and the corresponding parameters
print(gs.best_score_)    # Best accuracy score achieved during cross-validation
print(gs.best_params_)   # Best combination of hyperparameters

# Finally, we use the independent test dataset to estimate the performance of the best-selected model,
# which is available via the best_estimator_ attribute of the GridSearchCV object
# clf: classifier
clf = gs.best_estimator_
clf.fit(X_train, y_train)
print(f'Test accuracy: {clf.score(X_test, y_test):.3f}')
