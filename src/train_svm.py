from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def train_svm(X_train, y_train):
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf']
    }
    grid = GridSearchCV(SVC(probability=True, random_state=42), param_grid, cv=10, scoring='f1_macro', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_
