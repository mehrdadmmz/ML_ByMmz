from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.model_selection import train_test_split

# Load the dataset
data = load_iris()
X, y = data.data, data.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Sequential Feature Selector
# Perform SBS (Sequential Backward Selection)
sbs = SFS(knn, 
          k_features=2,                # Desired number of features
          forward=False,               # False = Backward Selection
          floating=False,              # No floating (standard SBS)
          scoring='accuracy',          # Use accuracy for scoring
          cv=5)                        # 5-fold cross-validation

# Fit SBS on the training data
sbs = sbs.fit(X_train, y_train)

# Get the selected feature indices
print(f"Selected feature indices: {sbs.k_feature_idx_}")
print(f"Selected feature names: {sbs.k_feature_names_}")
print(f"Feature performance (accuracy): {sbs.k_score_}")

# Transform the data to retain only selected features
X_train_sbs = sbs.transform(X_train)
X_test_sbs = sbs.transform(X_test)

# Train and evaluate on the reduced feature set
knn.fit(X_train_sbs, y_train)
accuracy = knn.score(X_test_sbs, y_test)
print(f"Accuracy on reduced feature set: {accuracy:.2f}")
