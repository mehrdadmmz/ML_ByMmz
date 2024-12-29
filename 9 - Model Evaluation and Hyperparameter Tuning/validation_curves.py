"""
Validation curves are a useful tool for improving the performance of a model by addressing issues such
as overfitting or underfitting.

Learning curves:   plotting training and test accuracies as a func of sample size
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

# Define the range of the hyperparameter 'C' (inverse of regularization strength) for Logistic Regression
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

# Compute training and validation scores for each value of the hyperparameter 'C'
# validation_curve evaluates model performance for different values of a specified hyperparameter
train_scores, test_scores = validation_curve(estimator=pipe_lr,                 
                                             X=X_train,                         
                                             y=y_train,
                                             param_name="logisticregression__C",  # Hyperparameter name ('C' in Logistic Regression)
                                             param_range=param_range,           # Range of values for 'C'
                                             cv=10                              # 10-fold cross-validation
                                            )

# Calculate the mean and standard deviation of training accuracy for each value of 'C'
train_mean = np.mean(train_scores, axis=1)  
train_std = np.std(train_scores, axis=1)   

# Calculate the mean and standard deviation of validation accuracy for each value of 'C'
test_mean = np.mean(test_scores, axis=1)   
test_std = np.std(test_scores, axis=1)    

# Plot the validation curve for training accuracy
plt.plot(param_range, 
         train_mean,               
         color='blue',             
         marker='o',               
         markersize=5,             
         label="Training accuracy")  
plt.fill_between(param_range, 
                 train_mean + train_std,  
                 train_mean - train_std,  
                 alpha=0.15,
                 color='blue')            

# Plot the validation curve for validation accuracy
plt.plot(param_range, 
         test_mean,                
         color='green',            
         linestyle='--',           
         marker='s',               
         markersize=5,
         label="Validation accuracy")  
plt.fill_between(param_range, 
                 test_mean + test_std,   
                 test_mean - test_std,   
                 alpha=0.15,             
                 color='green')          

# Add gridlines, axis labels, legend, and log scale for better visualization
plt.grid()                                
plt.xscale('log') # Use a logarithmic scale for the x-axis (hyperparameter 'C')
plt.legend(loc='lower right')             
plt.xlabel("Parameter C")
plt.ylabel("Accuracy")                    
plt.ylim([0.8, 1.0])                      
plt.show()
