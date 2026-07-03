from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

def train_dt(X_train, y_train):
    param_grid = {
        'max_depth': [None, 10, 20],
        'criterion': ['gini', 'entropy']
    }
    grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)
    grid.fit(X_train, y_train)
    return grid.best_estimator_
