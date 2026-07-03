def get_features_and_target(df, feature_cols, target_col):
    """Split dataframe into features and target."""
    X = df[feature_cols]
    y = df[target_col]
    return X, y

def get_recommended_features():
    """Return list of recommended features for Maternal Health Risk (including Pulse Pressure)."""
    return ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate', 'PulsePressure']
