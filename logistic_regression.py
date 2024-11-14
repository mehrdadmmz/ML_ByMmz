import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



class LogisticRegressionGD: 
    """
     Gradient descent-based logistic regression classifier.

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
          Weights after training.
        b_ : Scalar
          Bias unit after fitting.
        losses_ : list
           Log loss function values in each epoch.
           
    """
    def __init__(self, eta=0.1, n_iter=50, random_state=1): 
        self.eta = eta 
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit(self, X, y):
        """ Fit training data.

            Parameters
            ----------
            X : {array-like}, shape = [n_examples, n_features]
              Training vectors, where n_examples is the number of examples and
              n_features is the number of features.
            y : array-like, shape = [n_examples]
              Target values.

            Returns
            -------
            self : Instance of LogisticRegressionGD

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float64(0.0)
        self.losses_ = []
        
        for i in range(self.n_iter): 
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = ((-y.dot(np.log(output))) - ((1-y).dot(np.log(1-output)))) / X.shape[0]
            self.losses_.append(loss)
            
        return self
            
    def net_input(self, X): 
        """Calculate the net input"""
        return np.dot(X, self.w_) + self.b_
    
    def activation(self, z): 
        """Compute logistic sigmoid activation"""
        # Clip (limit) the values in an array. Given an interval, values 
        # outside the interval are clipped to the interval edges.
        return (1.0) / (1 + np.exp(-np.clip(z, -250, 250)))
    
    def predict(self, X): 
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)    
    
    
if __name__ == "__main__": 
        """
        when we fit a logistic reg model, we have to keep in mind that it only works 
        for binary classification tasks. So, we only consider setosa, and versicolor
        (class 0, 1)
        """
        iris = datasets.load_iris()
        X = iris.data[:100, [2, 3]]
        y = iris.target[:100]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                            test_size=0.3, 
                                                            random_state=1, 
                                                            stratify=y, )
        sc = StandardScaler()
        X_train_std = sc.fit_transform(X_train)
        X_test_std = sc.fit_transform(X_test)
        
        X_train_01_subset = X_train_std[(y_train == 0) | (y_train == 1)]
        y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]
        
        lrgd = LogisticRegressionGD(eta=0.3, 
                                    n_iter=1000, 
                                    random_state=1)
        
        lrgd.fit(X_train_01_subset, y_train_01_subset)
        plot_decision_regions(X = X_train_01_subset, 
                              y = y_train_01_subset, 
                              classifier=lrgd)
        
        plt.xlabel("Petal length [standardized]")
        plt.ylabel("Peal width [standardized]")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()
