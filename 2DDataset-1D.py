# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Create a sample 2D dataset
np.random.seed(42)
X = np.random.rand(30, 2) * 10  # 30 data points in 2D space

# Standardize the data
X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Perform PCA
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X_std)

# Display the steps and the resulting transformed data
print("Original Data (2D):")
print(X[:5, :])  # Displaying the first 5 rows for illustration

print("\nStandardized Data (2D):")
print(X_std[:5, :])

print("\nEigenvalues:")
print(pca.explained_variance_)

print("\nEigenvectors:")
print(pca.components_)

print("\nTransformed Data (1D):")
print(X_pca[:5, :])

# Visualize the original and transformed data
plt.scatter(X[:, 0], X[:, 1], label='Original Data')
plt.scatter(X_pca, np.zeros_like(X_pca), label='Transformed Data (1D)', marker='^')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title('PCA: 2D to 1D Transformation')
plt.show()
