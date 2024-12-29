# Randomized search CV 

# In randomized search, we draw hyperparameter
# configurations randomly from distributions (or discrete sets). In contrast to grid search, randomized
# search does not do an exhaustive search over the hyperparameter space.

# while RandomizedSearchCV can accept similar discrete lists of values as inputs for the parameter
# grid, which is useful when considering categorical hyperparameters, its main power lies in
# the fact that we can replace these lists with distributions to sample from.

import scipy
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Using a loguniform distribution instead of a regular uniform distribution
# ensures that the sampling of hyperparameters spans several orders of magnitude effectively.
# For example, in a uniform distribution over [0.0001, 1000.0], most samples would 
# cluster in the larger range (e.g., [10.0, 1000.0]) because the range is much wider.
# A loguniform distribution gives equal probability of sampling values from logarithmic 
# ranges like [0.0001, 0.001] and [10.0, 100.0], which is crucial for parameters like 'C' 
# and 'gamma' that often vary over multiple magnitudes in SVMs.
param_range = scipy.stats.loguniform(0.0001, 1000.0)

# Create a pipeline to standardize the features before feeding them into the SVM.
# StandardScaler ensures that the features have zero mean and unit variance, 
# which is essential for SVMs to perform optimally.
pipe_svc = make_pipeline(StandardScaler(), 
                         SVC(random_state=1))  

# Define the parameter grid for hyperparameter tuning.
param_grid = [
    # Tuning the 'C' parameter for a linear kernel.
    {'svc__C': param_range,                # 'C' controls the margin/regularization trade-off
     'svc__kernel': ['linear']},           # Kernel type is set to 'linear'

    # Tuning both 'C' and 'gamma' for the RBF kernel.
    # 'gamma' controls the influence of training points on the decision boundary.
    {'svc__C': param_range,                # 'C' is sampled from the loguniform distribution
     'svc__gamma': param_range,            # 'gamma' is also sampled from the loguniform distribution
     'svc__kernel': ['rbf']}               # Kernel type is set to 'rbf' (non-linear kernel)
]

# Perform randomized hyperparameter search using cross-validation.
# RandomizedSearchCV supports arbitrary distributions for sampling hyperparameters 
# (e.g., loguniform), allowing us to efficiently search over large spaces of values.
rs = RandomizedSearchCV(estimator=pipe_svc,            # The pipeline with SVM
                        param_distributions=param_grid, # Hyperparameter search space
                        scoring='accuracy',            # Evaluate models using accuracy
                        refit=True,                    # Automatically refit the best model
                        n_iter=20,                     # Number of hyperparameter combinations to try
                        cv=10,                         # 10-fold cross-validation
                        random_state=1,                # Set random state for reproducibility
                        n_jobs=-1)                     # Use all available CPU cores

# Fit the RandomizedSearchCV object on the training data.
rs = rs.fit(X_train, y_train)

# Print the best cross-validation accuracy score achieved during the search.
print(rs.best_score_)

# Print the combination of hyperparameters that resulted in the best score.
print(rs.best_params_)



