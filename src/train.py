import os
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Load custom core functions
from ml_core import load_data

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

def run_training_pipeline():
    print("=== Memulai Pipeline Pelatihan Modular ===")
    data_path = os.path.join(os.path.dirname(__file__), '../data/Maternal Health Risk Data Set.csv')
    
    # 1. Load data & engineer features
    df = load_data(data_path)
    
    # Encode target
    le = LabelEncoder()
    df['RiskLevel_Encoded'] = le.fit_transform(df['RiskLevel'])
    
    initial_features = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']
    features = initial_features + ['PulsePressure', 'MeanBP', 'ShockIndex']
    X = df[features]
    y = df['RiskLevel_Encoded']
    
    # 2. Split data stratified
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 4. Feature Selection
    selector = SelectKBest(score_func=mutual_info_classif, k=8)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    
    # 5. Class Imbalance Handling
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train_selected, y_train)
    
    # Train best model (Random Forest)
    print("Melatih model terbaik (Random Forest)...")
    rf_params = {'n_estimators': [100, 200], 'max_depth': [10, 20], 'random_state': [42]}
    rf_grid = GridSearchCV(RandomForestClassifier(), rf_params, cv=5, scoring='f1_macro')
    rf_grid.fit(X_train_res, y_train_res)
    best_rf = rf_grid.best_estimator_
    
    # Save best model pack
    model_dir = os.path.join(os.path.dirname(__file__), '../models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'best_model.joblib')
    
    export_data = {
        'model': best_rf,
        'scaler': scaler,
        'selector': selector,
        'le': le,
        'features': features
    }
    joblib.dump(export_data, model_path)
    print(f"Model berhasil dilatih dan disimpan di: {model_path}")

if __name__ == "__main__":
    run_training_pipeline()
