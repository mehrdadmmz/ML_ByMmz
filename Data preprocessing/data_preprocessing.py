import sys
import pandas as pd 
import numpy as np 
from io import StringIO
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

csv_data = """
A,B,C,D
1.0, 2.0, 3.0, 4.0
5.0,6.0,,8.0
10.0,11.0,12.0,
"""

df = pd.read_csv(StringIO(csv_data))

# other strategies will be median and most_frequent
imr = SimpleImputer(missing_values=np.nan, 
                    strategy="mean")
imr = imr.fit(df.values)
imputed_data = imr.transform(df.values)
imputed_data

# new dataframe with ordinal, nominal , and numercial features
# also it has non-ordinal class labels
df = pd.DataFrame([
    ["green", "M", 10.1, "class2"], 
    ["red", "L", 13.5, "class1"], 
    ["blue", "XL", 15.3, "class2"]])
df.columns = ["color", "size", "price", "classlabel"]

# mapping ordinal features
size_mapping = {"XL": 3, "L": 2, "M": 1}
df["size"] = df["size"].map(size_mapping)

# encoding class labels
class_mapping = {label: idx for idx, label in enumerate(np.unique(df["classlabel"]))}
df["classlabel"] = df["classlabel"].map(class_mapping)

X = df[["color", "size", "price"]].values

# class label encoder and decoder
class_le = LabelEncoder()
encoded_labels = class_le.fit_transform(df["classlabel"].values)
encoded_labels

decoded_labels = class_le.inverse_transform(encoded_labels)
decoded_labels


# performing one-hotencoding on color feature
# Color one-hot encoder
# unique values of color col will be sorted as blue, green, red in this order 
color_ohe = OneHotEncoder()
color_ohe.fit_transform(X[:, 0].reshape(-1, 1)).toarray(

# transforming columns in a multi-feature array
# modifying the first column feature (colors) and keeping the the other two columns the same as it is
c_transf = ColumnTransformer([
    ("onehot", OneHotEncoder(), [0]), 
    ("nothing", "passthrough", [1, 2]),
])
c_transf.fit_transform(X).astype(float)

# an easier way of doing this using pd.get_dummies() which will 
# conversation only string columns and leave all other columns unchanged
pd.get_dummies(df[["price", "color", "size"]])

