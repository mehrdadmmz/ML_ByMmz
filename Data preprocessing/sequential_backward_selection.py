from sklearn.base import clone
# Construct a new unfitted estimator with the same parameters.Clone does a deep copy of the model in an estimator without
# actually copying attached data. It returns a new estimator with the same parameters that has not been fitted on any data.
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
# combinations('ABCD', 2) ----> AB AC AD BC BD CD
from itertools import combinations

"""Start with the full feature set Xd, where d is the number of features. For each feature x ∈ Xk (current feature 
set), evaluate the criterion J(Xk−x). Remove the feature x that results in the smallest performance loss. (Basically 
the one with the highest accuracy score since the diff would be lower, meaning it contributed the least to the data, 
so it won't affect that much) Repeat until the desired number of features remain."""

"""Sequential Backward Selection"""


class SBS:
    """The __init__ method initializes the SBS class with parameters"""
    def __init__(self, estimator, k_features, scoring=accuracy_score, test_size=0.25, random_state=1):
        # A scoring function (default is accuracy_score from sklearn) used to evaluate model performance.
        self.scoring = scoring
        # estimator is an ML model. Clone function is used to make an identical copy of the
        # model to avoid changes to the original object.
        self.estimator = clone(estimator)
        # Desired number of features to retain at the end.
        self.k_features = k_features
        # The proportion of the dataset to be used as the test se
        self.test_size = test_size
        self.random_state = random_state

    """The fit method performs the Sequential Backward Selection"""
    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state)

        dim = X_train.shape[1]  # Number of features
        self.indices_ = tuple(range(dim))  # Start with all features
        self.subsets_ = [self.indices_]  # Subset history: full set initially
        score = self._calc_score(
            X_train, y_train, X_test, y_test, self.indices_)
        # List of scores, starting with full feature set score that we added as the first score in the list
        self.scores_ = [score]

        """Checking on all combinations of indices and keep track of the ones which we wanna remove later
        till we get the number of features we are looking for"""
        while dim > self.k_features:
            scores = []  # List to store scores for each candidate subset
            subsets = []  # List to store subsets of features

            """
            combinations generates subsets of size dim−1 at each iteration (removing one feature at a time).
            for x in combinations("012", 2): 
                print(x)
    
                ('0', '1')
                ('0', '2')
                ('1', '2')
            """
            for p in combinations(self.indices_, r=dim - 1):
                # calculating the score of each combination and then append them to the list to keep track of
                score = self._calc_score(X_train, y_train, X_test, y_test, p)
                scores.append(score)
                # appending the combination to the subsets list
                subsets.append(p)

            # best combination based on their associated score
            # np.argmax: Returns the indices of the maximum values along an axis.
            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])

        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        # we will only consider the indices we want to test (like a small subset of the full features)
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score
