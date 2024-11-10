from sklearn import datasaets 
import numpy as np 

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

print("Class labels: ", np.unique(y))
