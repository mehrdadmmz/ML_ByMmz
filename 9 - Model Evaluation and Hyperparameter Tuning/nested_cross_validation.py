"""
In nested cross-validation, we have an outer k-fold cross-validation loop to split the data into training
and test folds, and an inner loop is used to select the model using k-fold cross-validation on the training
fold. After model selection, the test fold is then used to evaluate the model performance.
"""

# Import necessary libraries
from sklearn.svm import SVC  # Support Vector Classifier
from sklearn.model_selection import GridSearchCV, cross_val_score  # For grid search and cross-validation
from sklearn.pipeline import make_pipeline  # For creating machine learning pipelines
from sklearn.preprocessing import StandardScaler  # For feature scaling
import numpy as np  # For numerical operations

# Nested cross-validation setup:
# Outer loop: Splits the data into training and test folds.
# Inner loop: Performs hyperparameter tuning using cross-validation on the training fold.

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

# Define the range of hyperparameters to test.
param_range = [0.0001, 0.001, 0.01, 0.1, 
               1.0, 10.0, 100.0, 1000.0]

# Define the parameter grid for hyperparameter tuning.
param_grid = [
    # Case 1: Tune the 'C' parameter for a linear kernel.
    {
        'svc__C': param_range,                # Regularization parameter to control margin
        'svc__kernel': ['linear']            # Linear kernel
    },

    # Case 2: Tune both 'C' and 'gamma' for the RBF kernel.
    {
        'svc__C': param_range,                # Regularization parameter
        'svc__gamma': param_range,            # RBF kernel coefficient
        'svc__kernel': ['rbf']               # RBF (non-linear) kernel
    }
]

# Set up GridSearchCV for hyperparameter tuning.
gs = GridSearchCV(estimator=pipe_svc, 
                  param_grid=param_grid, 
                  scoring='accuracy',        # Use accuracy as the evaluation metric
                  refit=True,                 # Refit the model with the best parameters
                  n_jobs=-1,                  # Utilize all CPU cores for faster computation
                  cv=2)                       # Perform 2-fold cross-validation for each parameter combination

# Evaluate the grid search using 5x2 cross-validation.
scores = cross_val_score(gs,                # GridSearchCV object for hyperparameter tuning
                         X_train,           # Training data
                         y_train,           # Training labels
                         scoring='accuracy', # Evaluate performance using accuracy
                         cv=5)              # 5-fold cross-validation

# Print the mean and standard deviation of cross-validation accuracy.
print(f'CV accuracy: {np.mean(scores):.3f}'
      f'+/- {np.std(scores):.3f}')
