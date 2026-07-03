from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def train_rf(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5]
    }
    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
    grid.fit(X_train, y_train)
    return grid.best_estimator_
