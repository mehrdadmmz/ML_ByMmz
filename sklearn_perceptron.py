from sklearn import datasaets 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np 
import pandas as pd 

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

print("Class labels: ", np.unique(y))
