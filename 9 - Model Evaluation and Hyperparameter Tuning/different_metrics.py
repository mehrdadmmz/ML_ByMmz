import numpy as np  # For numerical operations
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC  # Support Vector Classifier
from sklearn.pipeline import make_pipeline  # For creating machine learning pipelines
from sklearn.preprocessing import StandardScaler  # For feature scaling
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.metrics import matthews_corrcoef

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
y_pred = pipe_svc.predict(X_test)

pre_val = precision_score(y_true=y_test, y_pred=y_pred)
print(f'Precision: {pre_val:.3f}')

rec_val = recall_score(y_true=y_test, y_pred=y_pred)
print(f'Recall: {rec_val:.3f}')

f1_val = f1_score(y_true=y_test, y_pred=y_pred)
print(f'F1: {f1_val:.3f}')

mcc_val = matthews_corrcoef(y_true=y_test, y_pred=y_pred)
print(f'MCC: {mcc_val:.3f}')
