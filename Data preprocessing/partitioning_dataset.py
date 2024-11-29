import sys
import pandas as pd 
import numpy as np 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder

# Wine dataset from UCI, which has 178 wine examples and 13 features
df_wine = pd.read_csv(
    'https://archive.ics.uci.edu/ml/'
    'machine-learning-databases/wine/wine.data', 
    header= None, 
)

df_wine.columns = ['Class label', 'Alcohol',
                   'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium',
                   'Total phenols', 'Flavanoids',
                   'Nonflavanoid phenols',
                   'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines',
                   'Proline']
# we have 3 unique class labels : Class labels [1 2 3]
print('Class labels', np.unique(df_wine['Class label']))

X, y = df_wine.iloc[:, 1:], df_wine.iloc[:, 0]

# providing the class lebel array y as an argument to stratify enusres that both 
# training and tes datasets have the same class proportions as the original dataset
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y,
                                                    test_size= 0.3, 
                                                    random_state=0, 
                                                    stratify=y) 

# Normalization: MinMaxSclaer
# When features need to be bounded within a specific range.
# For models sensitive to the magnitude of features (e.g., Neural Networks, KNN).
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train) 
X_test_norm = mms.transform(X_test)

# Standardization: StandardScaler 
# For algorithms that assume or benefit from standardization (e.g., PCA, SVMs).
# When features are roughly Gaussian-distributed.
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train) 
X_test_std = stdsc.transform(X_test)

# Robust Sclaer: RobustSclaer
# When the dataset contains significant outliers.
# When you want a scaling method that minimizes the influence of extreme values.
rbstsc = RobustScaler()
X_train_rbst = rbstsc.fit_transform(X_train)
X_test_rbst = rbstsc.transform(X_test)
