"""Sequential backward/forward selection using mlxtend"""
import pandas as pd 
from mlxtend.feature_selection import SequentialFeatureSelector as SFS 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# URL of the dataset 
wine_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'

# Load the dataset
wine = pd.read_csv(wine_url, header=None)
wine.columns = ['Class label', 'Alcohol',
                'Malic acid', 'Ash',
                'Alcalinity of ash', 'Magnesium',
                'Total phenols', 'Flavanoids',
                'Nonflavanoid phenols',
                'Proanthocyanins',
                'Color intensity', 'Hue',
                'OD280/OD315 of diluted wines',
                'Proline']

# Split the dataset
X, y = wine.iloc[:, 1:], wine.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.3)

# Initialize the classifier
knn = KNeighborsClassifier(n_neighbors=5)

# Perform SBS (Sequential Backward Selection)
sbs =SFS(estimator=knn,
         k_features=3,        # Desired number of features
         forward=False,       # False = Backward Selection
         floating=False,      # No floating (standard SBS)
         scoring="accuracy",  # Use accuracy for scoring
         cv=5,                # 5-fold cross-validation
         )

# Perform SFS (Sequential Forward Selection)
# sfs = SFS(estimator=knn,
#          k_features=3,        # Desired number of features
#          forward=True,        # True = Forward Selection --> SFS starts with an empty set of features and 
#                               # adds features one by one based on their contribution to model performance.
#          floating=False,      # No floating (standard SBS)
#          scoring="accuracy",  # Use accuracy for scoring
#          cv=5,                # 5-fold cross-validation
#          )


# Fit SBS on the training data
sbs.fit(X_train, y_train)

# Get the selected feature indices
print(f"Selected feature indices: {sbs.k_feature_idx_}")
print(f"Selected feature names: {sbs.k_feature_names_}")
print(f"Selected features accuracy: {sbs.k_score_}")

# Transform the data to retain only selected features
X_train_sbs = sbs.transform(X_train)
X_test_sbs = sbs.transform(X_test)

# Train and evaluate on the reduced feature set
knn.fit(X_train_sbs, y_train)
accuracy = knn.score(X_test_sbs, y_test)
print(f"Accuracy on reduced feature set: {accuracy:.2f}")
