from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score
import numpy as np

# Load the wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Dataset loaded and split into train and test sets.')

clf = DecisionTreeClassifier(random_state=42)
param_dist = {
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

# Use RandomizedSearchCV to find the best hyperparameters
random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=100, cv=5, verbose=1, random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)

# Predict on the test set using the best model
y_pred = random_search.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)

print('Best hyperparameters:', random_search.best_params_)
print('Test set accuracy:', accuracy)


shuffle_split = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

# Step 6: Train 10 decision trees on each subset
random_forest = RandomForestClassifier(**random_search.best_params_)
accuracies = []
for train_index, _ in shuffle_split.split(X_train):
    X_subset, y_subset = X_train[train_index], y_train[train_index]
    dt_subset = DecisionTreeClassifier(**random_search.best_params_)
    dt_subset.fit(X_subset, y_subset)
    y_pred_subset = dt_subset.predict(X_test)
    accuracy_subset = accuracy_score(y_test, y_pred_subset)
    accuracies.append(accuracy_subset)


print(accuracies)