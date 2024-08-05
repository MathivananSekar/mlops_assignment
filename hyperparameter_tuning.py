from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Define model and hyperparameters
model = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [10, 20, 30],
    'max_depth': [5, 10, 15]
}

# Perform Grid Search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best parameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Evaluate best model
accuracy = best_model.score(X_test, y_test)

# Save the best model
joblib.dump(best_model, 'flask_app/best_model.joblib')

# Print results
print(f"Best Parameters: {best_params}")
print(f"Test Accuracy: {accuracy}")
