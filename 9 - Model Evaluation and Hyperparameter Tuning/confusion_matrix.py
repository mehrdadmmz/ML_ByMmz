# A confusion matrix is simply a square matrix that reports the counts of the true positive (TP), true negative
# (TN), false positive (FP), and false negative (FN) predictions of a classifier

from sklearn.svm import SVC  # Support Vector Classifier
from sklearn.pipeline import make_pipeline  # For creating machine learning pipelines
from sklearn.preprocessing import StandardScaler  # For feature scaling
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np  # For numerical operations


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
confmat = confusion_matrix(y_pred=y_pred, y_true=y_test)

# [[71  1]
#  [ 2 40]]
print(confmat)

fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.4)
for i in range(confmat.shape[0]): 
    for j in range(confmat.shape[1]): 
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
        
ax.xaxis.set_ticks_position('bottom')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show() 
