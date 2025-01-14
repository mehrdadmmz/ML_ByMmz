# make scorer: 

# We can use a different scoring metric than accuracy in the GridSearchCV via the scoring
# parameter. A complete list of the different values that are accepted by the scoring parameter can be
# found at http://scikit-learn.org/stable/modules/model_evaluation.html.
    
# Remember that the positive class in scikit-learn is the class that is labeled as class 1. If we want to
# specify a different positive label, we can construct our own scorer via the make_scorer function, which
# we can then directly provide as an argument to the scoring parameter in GridSearchCV (in this example,
# using the f1_score as a metric):

import pandas as pd 
import numpy as np 

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score

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

# Create a pipeline with feature scaling and an SVC model.
pipe_svc = make_pipeline(StandardScaler(), 
                         SVC(random_state=1)) 

pipe_svc.fit(X_train, y_train)

# positive class in scikit-learn is the class that is labeled as class 1, here we just wanna change it
scorer = make_scorer(f1_score, pos_label = 0) 

c_gamma_range = [0.01, 0.1, 1.0, 10.0]

param_grid = [{'svc__C': c_gamma_range, 
               'svc__kernel': ['linear']}, 
              
              {'svc__C':c_gamma_range,
               'svc__gamma': c_gamma_range, 
               'svc__kernel': ['rbf']}
             ]

gs = GridSearchCV(estimator=pipe_svc, 
                  param_grid=param_grid, 
                  scoring=scorer, 
                  cv=10, 
                  n_jobs=-1,)

gs = gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)
