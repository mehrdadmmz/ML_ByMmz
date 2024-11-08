import sys
import os 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap

class AdalineSGD: 
    """ADAptive LInear NEuron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    Shuffle: bool (default: True)
        shuffles training data every epoch if True 
        to prevent cycles. 
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
    
    def __init__(self, eta = 0.01, n_iter = 50, shuffle=True, random_state = None): 
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.w_initialized = False
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
        self._initialize_weights(X.shape[1])
        self.losses_ = []
        for i in range(self.n_iter): 
            if self.shuffle: 
                X, y = self._shuffle(X, y)
            losses = []
            for xi, target in zip(X, y): 
                losses.append(self._update_weights(xi, target))
            avg_loss = np.mean(losses)
            self.losses_.append(avg_loss)
        return self
        
    def partial_fit(self, X, y): 
        """Fit training data without reinitializing the weights"""
        if not self.w_initialized: 
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1: 
            for xi, target in zip(X, y): 
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self
    
    def _shuffle(self, X, y): 
        """Shuffle training data"""
        # generates a random permutation (reordering) of indices from 0 to len(y)-1.
        # returns a randomly shuffled array of indices.
        r = self.rgen.permutation(len(y))
        # X[r] and y[r] reorder the input features X and labels y 
        # based on the randomly shuffled indices stored in r.
        # we will basically passing a new shuffled array of indicies to 
        # X and y and they will be ordered based on the new indicies
        return X[r], y[r]
        
    
    def _initialize_weights(self, m):
        """Initialize weights to small random numbers"""
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ =  self.rgen.normal(loc=0.0, scale=0.01, size=m)
        self.b_ = np.float64(0.0)
        self.w_initialized = True
    
    def _update_weights(self, xi, target):
        """Apply adaline learning rule to update the weights"""
        output = self.activation(self.net_input(xi))
        error = target - output
        self.w_ += self.eta * 2.0 * xi * (error)
        self.b_ += self.eta * 2.0 * error
        loss = error ** 2
        return loss
    
    def net_input(self, X): 
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_
    
    def activation(self, X): 
        """Compute linear activation"""
        return X
    
    def predict(self, X): 
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
    
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
    # X is now a matrix 100*2 
    X = df.iloc[0:100, [0, 2]].values
    
    ada_sgd = AdalineSGD(eta=0.01, n_iter=15, random_state=1)
    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
    ada_sgd.fit(X_std, y)
    plot_decision_regions(X_std, y, classifier=ada_sgd)
    plt.title("Adaline - Stochastic Gradient Descent")
    plt.xlabel("Sepal Length [standardized]")
    plt.ylabel("Petal Length [standardized]")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()
    
    plt.plot(range(1, len(ada_sgd.losses_) + 1), 
            ada_sgd.losses_, 
            marker="o")
    plt.title("Average loss per epochs using stochastic gradient descent")
    plt.xlabel("Epochs")
    plt.ylabel("Average Loss")
    plt.tight_layout()
    plt.show()
