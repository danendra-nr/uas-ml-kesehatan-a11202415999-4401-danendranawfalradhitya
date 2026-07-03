import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, roc_auc_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_score


def evaluate_model(model, X_test, y_test, target_names=None):
    y_pred = model.predict(X_test)

    try:
        y_prob  = model.predict_proba(X_test)
        roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
    except (AttributeError, ValueError):
        roc_auc = None

    return {
        'accuracy':         accuracy_score(y_test, y_pred),
        'f1_macro':         f1_score(y_test, y_pred, average='macro'),
        'roc_auc':          roc_auc,
        'report':           classification_report(y_test, y_pred, target_names=target_names),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'predictions':      y_pred
    }


def cross_validate_model(model, X, y, n_splits=10, random_state=42):
    skf    = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
    return {
        'cv_scores': scores.tolist(),
        'cv_mean':   scores.mean(),
        'cv_std':    scores.std()
    }


def print_evaluation(name, metrics, cv_result=None):
    print(f"\n{'='*55}")
    print(f"  Evaluasi: {name}")
    print(f"{'='*55}")
    print(f"  Accuracy   : {metrics['accuracy']:.4f}")
    print(f"  F1 Macro   : {metrics['f1_macro']:.4f}")
    if metrics.get('roc_auc') is not None:
        print(f"  ROC-AUC    : {metrics['roc_auc']:.4f}")
    if cv_result:
        print(f"  CV Accuracy: {cv_result['cv_mean']:.4f} ± {cv_result['cv_std']:.4f}  (10-fold)")
    print("\n  Classification Report:")
    print(metrics['report'])
