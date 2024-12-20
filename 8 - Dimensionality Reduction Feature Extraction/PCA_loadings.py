"""
Sometimes, we are interested to know about how much each original
feature contributes to a given principal component. These contributions are often called loadings.
The factor loadings can be computed by scaling the eigenvectors by the square root of the eigenvalues.
The resulting values can then be interpreted as the correlation between the original features and
the principal component.

# In sklearn: 
# eigenvectors -- > pca.components_ 
# eigenvalues  -- > pca.explained_variance_

After plotting the bar we can see that: 
for example, Alcohol has a negative correlation with the first principal
component (approximately –0.3), whereas Malic acid has a positive correlation (approximately 0.54).
Note that a value of 1 describes a perfect positive correlation whereas a value of –1 corresponds to a
perfect negative correlation
"""

