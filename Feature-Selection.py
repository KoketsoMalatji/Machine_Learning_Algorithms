# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, chi2
import pandas as pd

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Convert the data to a DataFrame for better visualization
df = pd.DataFrame(X, columns=iris.feature_names)
df['Target'] = y

# Feature selection using SelectKBest with chi-squared test
k = 2  # Select the top 2 features
selector = SelectKBest(chi2, k=k)
X_new = selector.fit_transform(X, y)

# Get the selected features
selected_features = df.columns[:-1][selector.get_support()]

# Display the original and selected features
print("Original Features:")
print(df.columns[:-1])  # Exclude the target column
print("\nSelected Features:")
print(selected_features)


