# Import necessary libraries
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.ensemble import RandomForestClassifier

# Generate a dummy dataset (replace this with your dataset loading/preparation code)
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose your classification model
model = RandomForestClassifier()  # Replace with your chosen classification model

# Implement 5-fold cross-validation
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
accuracy_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy')

# Compute average accuracy
average_accuracy = sum(accuracy_scores) / num_folds

# Print the average accuracy
print(f"Average Accuracy: {average_accuracy:.4f}")


