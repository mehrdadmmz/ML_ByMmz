# Import necessary libraries
# Note: Enabling Halving Search CV is not required anymore in the latest version of scikit-learn.
# Uncomment the line below if using an older version where explicit enabling is necessary.
# from sklearn.experimental import enable_halving_search_cv  

import scipy  # For generating distributions for hyperparameter tuning
from sklearn.model_selection import HalvingRandomSearchCV, HalvingGridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline  # For creating machine learning pipelines
from sklearn.preprocessing import StandardScaler  # For feature scaling
from sklearn.svm import SVC  # Support Vector Classifier

# Define the range of hyperparameters using a log-uniform distribution.
param_range = scipy.stats.loguniform(0.0001, 1000.0)

# Create a pipeline with feature scaling and an SVC model.
pipe_svc = make_pipeline(StandardScaler(), 
                         SVC(random_state=1))  

# Define the parameter grid for hyperparameter tuning.
param_grid = [
    # Case 1: Tune the 'C' parameter for a linear kernel.
    {
        'svc__C': param_range,               # Regularization parameter
        'svc__kernel': ['linear']           # Use a linear kernel
    }, 
    
    # Case 2: Tune both 'C' and 'gamma' for the RBF kernel.
    # 'gamma' controls the influence of training points on the decision boundary.
    {
        'svc__C': param_range,               # Regularization parameter
        'svc__gamma': param_range,           # RBF kernel coefficient
        'svc__kernel': ['rbf']              # Use an RBF kernel
    }
]

# Configure Halving Random Search CV:
# - n_candidates='exhaust': Considers all hyperparameter configurations so that the maximum number of resources 
#   (e.g., training examples) are used in the final iteration.
# - resource='n_samples': Specifies the training set size as the resource to vary across iterations.
# - factor=1.5: Determines the proportion of candidates that move to the next round (e.g., ~66% survive each round).
# - random_state=1: Ensures reproducibility of results.
hs = HalvingRandomSearchCV(estimator=pipe_svc, 
                           param_distributions=param_grid, 
                           n_candidates='exhaust',  # Exhaustive search over all configurations
                           resource='n_samples',    # Training samples as the resource
                           factor=1.5,             # Reduction factor for candidates
                           random_state=1,          # Set random seed for reproducibility
                           n_jobs=-1)              # Utilize all CPU cores for faster computation

# Perform hyperparameter tuning with HalvingRandomSearchCV.
hs.fit(X_train, y_train)  # Fit the model on the training data

# Print the best score and the corresponding hyperparameters.
print(hs.best_score_)
print(hs.best_params_)

# Retrieve the best estimator and evaluate it on the test set.
clf = hs.best_estimator_  # Best model from hyperparameter search
print(f'Test accuracy: {hs.score(X_test, y_test):.3f}')
