import sys
import pandas as pd 
import numpy as np 
from io import StringIO
from sklearn.impute import SimpleImputer

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

# new dataframe with ordinal, non-ordinal, and numercial features
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

