from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

# Load Breast Cancer dataset
cancer = load_breast_cancer()
X_cancer, y_cancer = cancer.data, cancer.target

# Initialize RandomForestClassifier (you can replace this with other classifiers)
clf = RandomForestClassifier()

# Perform 5-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf, X_cancer, y_cancer, cv=kfold)

# Compute average accuracy
average_accuracy = scores.mean()
print("Average Accuracy:", average_accuracy)
