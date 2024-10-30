import sys
import os 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

class Perceptron: 
    
    """Perceptron classifier.
   
   
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
        
    errors_ : list
        Number of misclassifications (updates) in each epoch.
    """
    
    def __init__ (self, eta = 0.01, n_iter = 50, random_state = 1): 
        self.eta = eta 
        self.n_iter = n_iter
        self.random_state = random_state
        
    # Via the fit method, we initialize the bias self.b_ to an initial value 0 and the weights in self.w_ to
    # a vector, ℝ^m , where m stands for the number of dimensions (features) in the dataset.    
    def fit(self, X, y): 
        """Fit training data.
        
        
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
            Training vectors, where n_examples is the number of
            examples and n_features is the number of features.
        y : array-like, shape = [n_examples]
            Target values.
            
            
        Returns
        -------
        self : object
        
        """
        # random number generator
        rgen = np.random.RandomState(self.random_state)
        # loc: Mean(Mu) and scale: std deviation 
        # , which will be a normal distribution.  
        self.w_ = rgen.normal(loc=0.0 ,scale=0.01, 
                              size=X.shape[1])
        self.b_ = np.float64(0.)
        self.errors_ = []
        
        for _ in range(self.n_iter): 
            errors = 0 
            for xi, target in zip(X, y): 
                update = self.eta * (target - self.predict(xi))
                self.w_ = update * xi 
                self.b_ = update 
                errors += int(update!=0.0)
            self.errors_.append(errors)
        return self
                
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_
        
    def predict(self, X): 
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, 0)


if __name__ == "__main__": 
  s = 'https://archive.ics.uci.edu/ml/'\
    'machine-learning-databases/iris/iris.data'
  df = pd.read_csv(s, 
                 header=None, 
                 encoding="utf-8")
  
  # select setosa and versicolor
  y = df.iloc[0:100, 4].values
  y = np.where(y == "Iris-setosa", 0, 1)
  
  # extract sepal length and petal length
  X = df.iloc[0:100, [0, 2]].values
  
  plt.scatter(X[:50, 0], X[:50, 1],
              color="red", marker="o", label="Setosa")
  plt.scatter(X[50: 100, 0], X[50: 100, 1], 
              color="blue", marker="s", label="Versicolor")
  
  plt.xlabel('Sepal length [cm]')
  plt.ylabel('Petal length [cm]')
  plt.legend(loc='upper left')
  plt.show()

  # we can see that a linear decision boundary should be sufficient to separate setosa from versicolor flowers. Thus,
  # a linear classifier such as the perceptron should be able to classify the flowers in this dataset perfectly.
  ppn = Perceptron(eta=0.1, n_iter=10)
  ppn.fit(X, y)
  plt.plot(range(1, len(ppn.errors_) + 1), 
           ppn.errors_, marker="o")
  plt.xlabel("Epochs")
  plt.ylabel("Number of updates")
  plt.show()
