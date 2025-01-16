# CMPT 420 HW0
# Mehrdad Momeni Zadeh, Spring 2025, Profossor: Wuyang Chen, Simon Fraser University

# necessary libraries
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold


# loading data
data = np.load('data.npz')
X_train, X_test, y_train, y_test = data['Xtrain'], data['Xtest'], data['ytrain'], data['ytest']

degrees = [2, 14, 20]  # polynomial degrees

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()  # flattening plots

# plots a, b, c (different degrees)
for i, degree in enumerate(degrees):
    ax = axes[i]

    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    lr = LinearRegression()
    lr.fit(X_train_poly, y_train)

    y_pred_train = lr.predict(X_train_poly)
    y_pred_test = lr.predict(X_test_poly)

    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    print(
        f"Degree= {degree} --> Train MSE={mse_train:.2f}, Test MSE={mse_test:.2f}")

    ax.scatter(X_train, y_train, color='blue', alpha=0.7,
               label='Train data')  # Scatter train data

    ax.plot(X_test,
            y_pred_test,
            'g-',
            linewidth=2,
            label=f"Poly degree{degree}",)

    ax.set_ylim(-10, 15)
    ax.set_title(f"degree = {degree}")
    ax.legend(loc='best')

# plot d (mse vs. degrees)
all_degrees = range(1, 16)
train_mses, test_mses = [], []

for degree in all_degrees:
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    lr = LinearRegression()
    lr.fit(X_train_poly, y_train)

    y_pred_train = lr.predict(X_train_poly)
    y_pred_test = lr.predict(X_test_poly)

    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)

    train_mses.append(mse_train)
    test_mses.append(mse_test)

ax_mse = axes[-1]
ax_mse.plot(all_degrees, train_mses, '-s', color='blue', label='train')
ax_mse.plot(all_degrees, test_mses,  '-x', color='green',  label='test')
ax_mse.set_xlabel("degree")
ax_mse.set_ylabel("mse")
ax_mse.set_title("mse vs. polynomial degree")
ax_mse.legend()

plt.tight_layout()
plt.show()


# KFold for 4-fold CV for finding and reporting the best degree
kfold = KFold(n_splits=4, shuffle=True, random_state=42)

all_degrees = range(1, 16)
cv_mses = []

for degree in all_degrees:
    poly = PolynomialFeatures(degree=degree)

    # We dont fit_transform on X_test for CV, because CV only uses the training set.
    X_train_poly = poly.fit_transform(X_train)

    lr = LinearRegression()

    scores = cross_val_score(lr,
                             X_train_poly,
                             y_train,
                             cv=kfold,
                             scoring='neg_mean_squared_error')

    mean_mse = -np.mean(scores)
    cv_mses.append(mean_mse)

# the degree that gives the smallest mean CV MSE
best_degree = all_degrees[np.argmin(cv_mses)]
print(f"Best degree via 4-fold cross-validation is {best_degree} with the MSE of {cv_mses[3]:.2f}")

# Best degree via 4-fold cross-validation is 4 with the MSE of 3.81
