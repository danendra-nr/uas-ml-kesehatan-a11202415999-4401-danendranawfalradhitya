import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif

def engineer_features(df):
    """
    Menambahkan fiturPulsePressure, MeanBP, dan ShockIndex ke dalam DataFrame.
    """
    df_copy = df.copy()
    df_copy['PulsePressure'] = df_copy['SystolicBP'] - df_copy['DiastolicBP']
    df_copy['MeanBP']        = (df_copy['SystolicBP'] + 2 * df_copy['DiastolicBP']) / 3
    df_copy['ShockIndex']    = df_copy['HeartRate'] / df_copy['SystolicBP'].replace(0, 1)
    return df_copy

def load_data(filepath):
    """
    Membaca dataset dan melakukan feature engineering.
    """
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    df = engineer_features(df)
    return df
