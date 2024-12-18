# Import necessary libraries
import pandas as pd  # For data loading and manipulation
import numpy as np   # For numerical operations
import matplotlib.pyplot as plt  # For visualization
from sklearn.model_selection import train_test_split  # For splitting the dataset
from sklearn.metrics import accuracy_score           # For evaluating accuracy
from sklearn.ensemble import RandomForestClassifier  # For building the Random Forest model
from sklearn.feature_selection import SelectFromModel # For selecting certain number of important features


# ------------------------------
# STEP 1: Load the Wine Dataset
# ------------------------------

# URL of the dataset (UCI Wine Dataset)
wine_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'

# Load the dataset into a DataFrame
# header=None ensures that there is no header row in the file
df_wine = pd.read_csv(wine_url, header=None)

# Assign column names to the DataFrame
df_wine.columns = ['Class label', 'Alcohol',
                   'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium',
                   'Total phenols', 'Flavanoids',
                   'Nonflavanoid phenols',
                   'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines',
                   'Proline']

# ------------------------------
# STEP 2: Split the Dataset
# ------------------------------

# Separate features (X) and target (y)
# Features: All columns except the first (Class label)
# Target: First column (Class label)
X, y = df_wine.iloc[:, 1:], df_wine.iloc[:, 0]

# Split the data into training and test sets (70% training, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1)

# Save feature names for later use
feat_labels = df_wine.columns[1:]

# ------------------------------
# STEP 3: Train the Random Forest Classifier
# ------------------------------

# Initialize a Random Forest Classifier with 500 trees (n_estimators)
# Random state ensures reproducibility
forest = RandomForestClassifier(n_estimators=500, random_state=1)


# ------------------------------
# STEP 4: Feature Importance
# ------------------------------

# Extract feature importances from the trained model
importances = forest.feature_importances_

# Sort the feature importances in descending order and get their indices
# np.argsort() returns the indices that would sort an array
# [::-1] reverses the order to descending
indices = np.argsort(importances)[::-1]

# Use SelectFromModel to select features based on importance scores
# Only features with importance >= 0.1 will be retained
sfm = SelectFromModel(forest, threshold=0.1, prefit=True)

# Transform the training data to keep only the selected features
X_selected = sfm.transform(X_train)

# Print the number of features that meet the threshold
print(f"Number of features that meet this threshold criterion: {X_selected.shape[1]}")

# Display the selected feature names and their importance scores
for f in range(X_selected.shape[1]):
    print(f"{f + 1:2d}) {feat_labels[indices[f]]:30} {importances[indices[f]]:.6f}")

