# Import necessary libraries
import pandas as pd  # For data loading and manipulation
import numpy as np   # For numerical operations
import matplotlib.pyplot as plt  # For visualization
from sklearn.model_selection import train_test_split  # For splitting the dataset
from sklearn.metrics import accuracy_score           # For evaluating accuracy
from sklearn.ensemble import RandomForestClassifier  # For building the Random Forest model

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

# Train the model on the training data
forest.fit(X_train, y_train)

# ------------------------------
# STEP 4: Feature Importance
# ------------------------------

# Extract feature importances from the trained model
importances = forest.feature_importances_

# Sort the feature importances in descending order and get their indices
# np.argsort() returns the indices that would sort an array
# [::-1] reverses the order to descending
indices = np.argsort(importances)[::-1]

# Print feature ranking
print("Feature ranking based on importance:\n")
for rank, idx in enumerate(indices, start=1):
    feature_name = feat_labels[idx]
    importance_score = importances[idx]
    print(f"{rank:2d}) {feature_name:30} {importance_score:.6f}")

# ------------------------------
# STEP 5: Plot Feature Importances
# ------------------------------

# Create a bar chart to visualize feature importance
plt.title("Feature Importance (Random Forest)")
plt.bar(range(X_train.shape[1]),        # X-axis: Feature indices
        importances[indices],           # Y-axis: Feature importance values
        align="center")                 # Align bars to center

# Add feature names to X-axis ticks
plt.xticks(range(X_train.shape[1]), 
           feat_labels[indices],        # Use sorted feature labels
           rotation=90)                 # Rotate labels for better visibility

# Set X-axis limits
plt.xlim(-1, X_train.shape[1])

# Improve layout and display the plot
plt.tight_layout()
plt.show()
