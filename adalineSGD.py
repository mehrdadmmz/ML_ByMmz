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
        pass
        
    def partial_fit(self, X, y): 
        pass
    
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
        self.b_ = np.float(0.0)
        self.w_initialized = True
    
    def _update_weights(self, xi, target): 
        pass
    
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
    pass
