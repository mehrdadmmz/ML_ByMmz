"""
Here, we will discuss how we can use learning curves to diagnose whether a learning algorithm
has a problem with overfitting (high variance) or underfitting (high bias).

Learning curves:   plotting training and test accuracies as a func of sameple size
Validation curves: we vary the values of the model parameters like parameter C in logistic reg 
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, learning_curve

# Loading the Breast Cancer Wisconsin dataset 
df = pd.read_csv('https://archive.ics.uci.edu/ml/'
                 'machine-learning-databases'
                 '/breast-cancer-wisconsin/wdbc.data',
                 header=None)

X, y = df.loc[:, 2:].values, df.loc[:, 1]

# label encoding 
le = LabelEncoder()

# After encoding the class labels (diagnosis) in an array, y, the malignant tumors are now represented
# as class 1, and the benign tumors are represented as class 0, respectively
y = le.fit_transform(y) 

# Splitting data 
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    stratify=y, 
                                                    random_state=1)

# Pipeline setup: StandardScaler for feature scaling followed by Logistic Regression
pipe_lr = make_pipeline(StandardScaler(), 
                        LogisticRegression(penalty='l2',      # L2 regularization to prevent overfitting
                                           max_iter=10_000))  # Max iterations to ensure convergence

# Generate training sizes: 10 equally spaced values between 10% and 100% of the training dataset size
# learning_curve evaluates model performance for increasing training dataset sizes
train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_lr,             
                                                        X=X_train,                     
                                                        y=y_train,                     
                                                        train_sizes=np.linspace(0.1, 1.0, 10),  # Sizes of training subsets
                                                        cv=10,                         # 10-fold cross-validation
                                                        n_jobs=1                       # Use a single CPU core
                                                    )

# Calculate the mean and standard deviation of training accuracy for each training size
train_mean = np.mean(train_scores, axis=1)  
train_std = np.std(train_scores, axis=1)   

# Calculate the mean and standard deviation of validation accuracy for each training size
test_mean = np.mean(test_scores, axis=1)   
test_std = np.std(test_scores, axis=1)    

# Plot the learning curve for training accuracy
plt.plot(train_sizes, 
         train_mean,               
         color='blue',             
         marker='o',               
         markersize=5,             
         label='Training accuracy')  
plt.fill_between(train_sizes, 
                 train_mean - train_std,  
                 train_mean + train_std,  
                 alpha=0.15,              
                 color='blue')         

# Plot the learning curve for validation accuracy
plt.plot(train_sizes, 
         test_mean,
         color='green',            
         linestyle='--',           
         marker='s',               
         markersize=5,             
         label='Validation accuracy')  
plt.fill_between(train_sizes, 
                 test_mean - test_std,   
                 test_mean + test_std,   
                 alpha=0.15,             
                 color='green')          

# Add gridlines, axis labels, and legend for better readability
plt.grid()                              
plt.xlabel('Number of training examples')  
plt.ylabel('Accuracy')                 
plt.legend(loc='lower right')          
plt.ylim([0.8, 1.03])
plt.show()
