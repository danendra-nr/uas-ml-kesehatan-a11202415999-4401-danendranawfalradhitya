import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False


def load_and_preprocess(filepath, apply_smote=True, test_size=0.2, random_state=42):
    df = pd.read_csv(filepath, encoding='utf-8-sig')

    df['PulsePressure'] = df['SystolicBP'] - df['DiastolicBP']
    df['MeanBP']        = (df['SystolicBP'] + 2 * df['DiastolicBP']) / 3
    df['ShockIndex']    = df['HeartRate'] / df['SystolicBP'].replace(0, 1)

    le = LabelEncoder()
    df['RiskLevel_Encoded'] = le.fit_transform(df['RiskLevel'])

    features = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp',
                'HeartRate', 'PulsePressure', 'MeanBP', 'ShockIndex']
    X = df[features]
    y = df['RiskLevel_Encoded']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    if apply_smote and SMOTE_AVAILABLE:
        sm = SMOTE(random_state=random_state)
        X_train_scaled, y_train = sm.fit_resample(X_train_scaled, y_train)
    elif apply_smote and not SMOTE_AVAILABLE:
        print("[WARNING] imbalanced-learn tidak terinstall. SMOTE dilewati.")

    return X_train_scaled, X_test_scaled, y_train, y_test, le, scaler, features
