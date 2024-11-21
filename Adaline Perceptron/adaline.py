import sys
import os 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap

class AdalineGD: 
    """ADAptive LInear NEuron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.


    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    b_ : Scalar
      Bias unit after fitting.
    losses_ : list
      Mean squared eror loss function values in each epoch.

    """
    
    def __init__(self, eta = 0.01, n_iter = 50, random_state = 1): 
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit(self, X, y): 
        """Fit training data.
        
        Parameters
        -----------
        X: {array_like}, shape = [n_examples, n_features]
            Training vectors, where n examples is the number
            of examples and n_features is the number of features. 
        y: array_like, shape = [n_examples]
            Target values. 
        
        Returns
        -----------
        self: object 
        
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, 
                              size=X.shape[1])
        self.b_ = np.float64(0.0)
        self.loses_ = []
        
        for i in range(self.n_iter): 
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = (errors**2).mean()
            self.loses_.append(loss)
        return self
    
    def net_input(self, X): 
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_
    
    def activation(self, X): 
        """Compute linear activation"""
        return X
    
    def predict(self, X): 
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)

if __name__ == "__main__": 
    try: 
        s = 'https://archive.ics.uci.edu/ml/'\
        'machine-learning-databases/iris/iris.data'
        print("From url: ", s)
        df = pd.read_csv(s, 
                         header=None, 
                         encoding="utf-8")
        
    except HTTPError: 
        s = 'iris.data'
        prtin("From local Iris path: ", s)
        df = pd.read_csv(s, 
                         header=None, 
                         encoding="utf-8")

    # select setosa and versicolor
    y = df.iloc[0:100, 4].values
    y = np.where(y == "Iris-setosa", 0, 1)
    
    # extract sepal length and petal length
    # X is now a matrix 100*2 
    X = df.iloc[0:100, [0, 2]].values


    # returns figure and Axes or array of Axes
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    
    ada1 = AdalineGD(n_iter=15, eta=0.1).fit(X, y)
    ax[0].plot(range(1, len(ada1.loses_) + 1), 
               np.log10(ada1.loses_), 
               marker="o")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("log(Mean squared error)")
    ax[0].set_title("Adaline - Learning rate 0.1")
    
    ada2 = AdalineGD(n_iter=15, eta=0.0001).fit(X, y)
    ax[1].plot(range(1, len(ada2.loses_) + 1), 
               np.log10(ada2.loses_), 
               marker="o")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("log(Mean squared error)")
    ax[1].set_title("Adaline - Learning rate 0.0001")
    
    plt.show()
