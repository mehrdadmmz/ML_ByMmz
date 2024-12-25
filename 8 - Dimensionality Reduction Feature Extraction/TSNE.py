import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE

""" t-distributed stochastic neighbour embedding (t-SNE): 
non-linear dimensionality reduction technique. TSNE is best for Visualizing high-dimensional data in 2D or 3D. 
Like PCA, t-SNE is an unsupervised method. 

The development and application of nonlinear dimensionality reduction techniques is also often
referred to as manifold learning, where a manifold refers to a lower dimensional topological space
embedded in a high-dimensional space.

t-SNE learns to embed data points into a lower-dimensional space
such that the pairwise distances in the original space are preserved. 

Since it projects the points directly (unlike PCA, it does not involve a projection matrix), 
we cannot apply t-SNE to new data points.
"""


digits = load_digits()

fig, ax = plt.subplots(1, 4)
for i in range(4):
    ax[i].imshow(digits.images[i], cmap='Greys') # Display data as an image

plt.show() # shows the first 4 images

X_digits = digits.data # (1797, 64) pixel values of each ex ranging from 0(completely black) to 16(completely white)
y_digits = digits.target # the actual numbers 0-9

# init='pca', which initializes the t-SNE embedding using PCA
tsne = TSNE(n_components=2, init='pca', random_state=123)

# projecting 64-dim dataset onto 2-dim space
X_digits_tsne = tsne.fit_transform(X_digits)

# This function visualizes the reduced 2D data (x) using different colors for each digit label (colors).
def plot_projection(x, colors): 
    
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')

    # Iterates over the digit classes (0–9)
    for i in range(10): 
        plt.scatter(x[colors == i, 0], x[colors == i, 1]) # Plots these points in the 2D space.
    
    for i in range(10):
        # Computes the median position of all points for a given digit class in 2D space
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        # Places the digit label (str(i)) at the median position for better visibility
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        # Adds a white outline around the text labels for better contrast with the background.
        # PathEffects.Stroke: Creates the outline.
        # PathEffects.Normal: Ensures the text is displayed normally.
        txt.set_path_effects([PathEffects.Stroke(linewidth=5, foreground='w'), 
                              PathEffects.Normal()])
        
plot_projection(X_digits_tsne, y_digits)
plt.show()
