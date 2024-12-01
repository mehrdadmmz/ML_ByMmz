import sys
import pandas as pd 
import numpy as np 
from io import StringIO
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer

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
print('Class labels', np.unique(df_wine['Class label']))

X, y = df_wine.iloc[:, 1:], df_wine.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y,
                                                    test_size= 0.3, 
                                                    random_state=0, 
                                                    stratify=y)

# Standardization: StandardScaler 
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train) 
X_test_std = stdsc.transform(X_test)

# l1 regularization 
# Note that C=1.0 is the default. You can increase
# or decrease it to make the regularization effect
# stronger or weaker, respectively.
lr = LogisticRegression(penalty='l1',
                        C=1.0,
                        solver='liblinear', 
                        multi_class='ovr',)

lr.fit(X_train_std, y_train)
print("Training accuracy: ", lr.score(X_train_std, y_train))
print("Test accuracy: ", lr.score(X_test_std, y_test))

# bias array
# Since we fit the LogisticRegression object on a multiclass dataset via the one-versus-rest (OvR)
# approach, the first intercept belongs to the model that fits class 1 versus classes 2 and 3, the second
# value is the intercept of the model that fits class 2 versus classes 1 and 3, and the third value is the
# intercept of the model that fits class 3 versus classes 1 and 2.
print(lr.intercept_)

# weight array 
# The weight array that we accessed via the lr.coef_ attribute contains three rows of weight coefficients,
# one weight vector for each class. Each row consists of 13 weights, where each weight is multiplied by
# the respective feature in the 13-dimensional Wine dataset to calculate the net input: 
# z = w1x1 + w2x2 + ... + wmxm = sum(xjwj) from j=1 to j=m and then + b 
print(lr.coef_)
