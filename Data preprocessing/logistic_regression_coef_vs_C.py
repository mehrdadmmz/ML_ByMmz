import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Wine dataset from UCI, which has 178 wine examples and 13 features
df_wine = pd.read_csv(
    'https://archive.ics.uci.edu/ml/'
    'machine-learning-databases/wine/wine.data', 
    header= None, 
)

# setting up the columns
df_wine.columns = ['Class label', 'Alcohol',
                   'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium',
                   'Total phenols', 'Flavanoids',
                   'Nonflavanoid phenols',
                   'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines',
                   'Proline']

X, y = df_wine.iloc[:, 1:], df_wine.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y,
                                                    test_size= 0.3, 
                                                    random_state=0, 
                                                    stratify=y)

stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

fig = plt.figure()
ax = plt.subplot(111)
colors = ['blue', 'green', 'red', 'cyan',
          'magenta', 'yellow', 'black',
          'pink', 'lightgreen', 'lightblue',
          'gray', 'indigo', 'orange']

weights, params = [], []

# As we will see, all feature weights will be zero if we penalize the model with a strong 
# regularization parameter (C < 0.01); C is the inverse of the regularization parameter, 𝜆
for c in np.arange(-4.0, 6.0):
    lr = LogisticRegression(penalty="l1", 
                            C=10.0 ** c, 
                            solver="liblinear", 
                            multi_class="ovr", 
                            random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1]) # here we care about the second class 
    params.append(10**c) # storing the value of C that we applied to 

weights = np.array(weights)
for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, 
             weights[:, column], 
             label=df_wine.columns[column + 1], # adding 1 since the first col name is just class label
             color=color,)
    
plt.axhline(0, color="black", linestyle="--", linewidth=3)
plt.xlim(10**-5, 10**5)
plt.ylabel("Weight coefficients")    
plt.xlabel("C(inverse regularization strength)")
plt.xscale('log')
plt.legend(loc="upper left")
ax.legend(loc="upper center", 
          bbox_to_anchor=(1.38, 1.03), 
          ncol=1, 
          fancybox=True)
plt.show()
