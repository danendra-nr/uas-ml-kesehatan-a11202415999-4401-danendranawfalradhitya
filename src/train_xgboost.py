from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

def train_xgboost(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
    }
    xgb = XGBClassifier(random_state=42, eval_metric='mlogloss', use_label_encoder=False)
    grid = GridSearchCV(xgb, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_
