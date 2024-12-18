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
                self.w_ += update * xi 
                self.b_ += update 
                errors += int(update!=0.0)
            self.errors_.append(errors)
        return self
                
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_
        
    def predict(self, X): 
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, 0)

# implementing a small convinience function to visualize the decision 
# boundaries for two-dimensional datasets. 

def plot_decision_regions(X, y, classifier, resolution=0.02): 
    # setup marker generator and color map 
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    # we need the same number of colors as the number of unique classes we have
    cmap = ListedColormap(colors[:len(np.unique(y))]) 

    # plot the decision surface
    # By subtracting 1 from the minimum and adding 1 to the maximum, we slightly expand the range. 
    # This ensures that the decision boundary isn't plotted right at the edge of your data points, 
    # which provides better visualization and prevents data points from being plotted on the edge of the graph.
    # min and max of the two features we are dealing with
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # np.arange creates a 1D array of values from x1_min to x1_max, spaced by resolution 0.02     
    # meshgrid takes 2, 1D array and returns 2, 2D matrices that represent all combinations of 
    # the input arrays 
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    # .ravel return a contiguous flattened array 
    # new way of doing it is using .reshape(-1)
    # 
    # stacks these 1D arrays as columns to create an array of coordinates
    # basically we have xx1, xx2, we flatten both then stack them on top 
    # of each other and the transpose it and pasa it to the predict function 
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    # Setting Plot Limits
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # This loop iterates over each unique class label in y 
    # idx is the index of the class, and cl is the class label itself
    for idx, cl in enumerate(np.unique(y)): 
        plt.scatter(x = X[y == cl, 0], 
                    y = X[y == cl, 1], 
                    alpha = 0.8, 
                    c = colors[idx],
                    marker = markers[idx],
                    label = f'Class {cl}', 
                    edgecolor = 'black', 
                   )    


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

# plotting decision boundaries using contourf 
plot_decision_regions(X, y, classifier=ppn)
plt.xlabel("Sepal length [cm]")
plt.ylabel("Petal length [cm]")
plt.legend(loc="upper left")
plt.show()
