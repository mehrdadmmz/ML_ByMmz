import numpy as np 
import matplotlib.pyplot as plt

"""
[wMAP, bMAP] (Maximum A Posteriori estimation).
"""
def omega_map_estimator(x, y, sigma_w_sq=1.0, sigma_b_sq=1.0):
    # make both x and y 1D array 
    x = x.flatten()
    y = y.flatten()
    n_samples = x.shape[0]
    
    # Reshape x and y to column vectors
    x = x.reshape(n_samples, 1)  
    y = y.reshape(n_samples, 1)
    
    # Add a column of ones to x for the intercept term
    X = np.hstack([x, np.ones((n_samples, 1))])  # now in the shape of (n_samples, 2)
    # Now, the columns of X correspond to w and b respectively
    
    # Prior precision (lambda) for w and b
    lambda_w = 1.0 / sigma_w_sq
    lambda_b = 1.0 / sigma_b_sq
    
    # Regularization matrix (Lambda)
    Lambda = np.array([[lambda_w, 0],
                       [0, lambda_b]])  # Shape: (2, 2)
    
    # Compute X^T X (LHS)
    A = X.T @ X 
    
    # Compute X^T y (RH)
    B = X.T @ y 
    
    # Compute inside parentheses (X^T X + Lambda)
    XtX_plus_Lambda = A + Lambda  # Shape: (2, 2)
    
    # Solve for omega_MAP: (X^T X + Lambda)^{-1} X^T y
    omega_MAP = np.linalg.solve(XtX_plus_Lambda, B)  # Shape: (2, 1)
    
    return omega_MAP

if __name__ == "__main__":       
    # The data source: Database of standard rated games, played on lichess.org
    # https://database.lichess.org/standard/lichess_db_standard_rated_2013-03.pgn.zst
    # x in the above data are ELO differences for White ELO - Black ELO
    # y is the win probability sampled from a probabilistic model from Liches DB

    # The x and y dimensions in this dataset
    x = np.array([[-139.15341508, -681.19390135,  374.83381147,  461.45761931,
           -205.65507596,    7.52847535, -551.03494308,  659.15105943,
           -215.83519299, -341.60486674,  590.2827852 ,  648.04672811,
            393.6312089 ,  190.73955575,  388.96338453, -252.161518  ,
           -251.64298135, -657.19690255, -352.94565567, -178.58059633,
            304.90248582]])
    y = np.array([[0.44610327, 0.1444    , 0.7737475 , 0.81391054, 0.24666655,
       0.49943957, 0.12402231, 0.93239468, 0.28009304, 0.14202444,
       0.88360894, 0.84497416, 0.72092098, 0.6790719 , 0.84907639,
       0.24405219, 0.27233288, 0.13289914, 0.25335661, 0.34657246,
       0.78085279]])
    
    # Mean values of the parameters from the MCMC samples
    linear_beta_mean = 0.00093696                           
    linear_intercept_mean = 0.51848817                      
    
    # Generate Elo differences over a range for plotting the sigmoid curve
    elo_diff_range = np.linspace(x.min(), x.max(), 100)
    
    x_true_plot = elo_diff_range
    y_true_plot = linear_beta_mean * elo_diff_range + linear_intercept_mean
    
    # Plot the synthetic data as a scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.5, s=10, color="blue", label="Simulated  Data")
    
    # Plot the "true" sigmoid model as a red, dashed line
    plt.plot(x_true_plot, y_true_plot, color="red", linestyle="--", label="Linear Model (Mean Params)")
    
    # Labels and legend
    plt.xlabel("Elo Difference")
    plt.ylabel("Win Probability")
    plt.title("Win Probability vs Elo Difference with Linear Model")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    w = omega_map_estimator(x, y)
    b_map, w_map = w[0], w[1]
    print(b_map)
    print(w_map)