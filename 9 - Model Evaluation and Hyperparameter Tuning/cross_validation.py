# holdout cross-validation 
# k-fold cross-validation 
# stratified k-fold cross-validation 

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

"""We will be working with the Breast Cancer Wisconsin dataset, which contains 569 examples
of malignant and benign tumor cells. The first two columns in the dataset store the unique ID
numbers of the examples and the corresponding diagnoses (M = malignant, B = benign), respectively.
Columns 3-32 contain 30 real-valued features that have been computed from digitized images of the
cell nuclei, which can be used to build a model to predict whether a tumor is benign or malignant."""

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
''' When we execute the fit method on the pipe_lr pipeline in the following code example,
StandardScaler first performed fit and transform calls on the training data. Second, the transformed
training data was passed on to the next object in the pipeline, PCA. Similar to the previous
step, PCA also executed fit and transform on the scaled input data and passed it to the final element
of the pipeline, the estimator.
Finally, the LogisticRegression estimator was fit to the training data after it underwent transformations
via StandardScaler and PCA. Again, we should note that there is no limit to the number of
intermediate steps in a pipeline; however, if we want to use the pipeline for prediction tasks, the last
pipeline element has to be an estimator. '''
pipe_lr = make_pipeline(StandardScaler(), 
                            PCA(n_components=2), 
                            LogisticRegression())

pipe_lr.fit(X_train, y_train)
y_pred = pipe_lr.predict(X_test)
test_acc = pipe_lr.score(X_test, y_test)
print(f"Test accuracy: {test_acc:.3f}")

# Performing stratified k-fold cross-validation 
kfold = StratifiedKFold(n_splits=10).split(X_train, y_train)
scores = cross_val_score(estimator=pipe_lr, 
                         X=X_train, 
                         y=y_train, 
                         cv=10, 
                         n_jobs=1)

print(f'CV accuracy scores" {scores}')
print(f"\nCV accuracy: {np.mean(scores):.3f} +/- {np.std(scores):.3f}")
