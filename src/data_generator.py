import pandas as pd
import numpy as np

def generate_synthetic_data(num_samples=100):
    """
    Menghasilkan data sintetis pasien hamil untuk keperluan pengujian.
    """
    np.random.seed(42)
    age = np.random.randint(15, 45, num_samples)
    sys_bp = np.random.randint(90, 150, num_samples)
    dia_bp = np.random.randint(60, 95, num_samples)
    bs = np.random.uniform(5.5, 15.0, num_samples)
    body_temp = np.random.uniform(97.8, 101.5, num_samples)
    heart_rate = np.random.randint(60, 100, num_samples)
    
    # Simple rule for risk level mapping
    risks = []
    for i in range(num_samples):
        score = 0
        if sys_bp[i] > 130 or dia_bp[i] > 85:
            score += 2
        if bs[i] > 10.0:
            score += 3
        if age[i] > 35:
            score += 1
            
        if score >= 4:
            risks.append('high risk')
        elif score >= 2:
            risks.append('mid risk')
        else:
            risks.append('low risk')
            
    df = pd.DataFrame({
        'Age': age,
        'SystolicBP': sys_bp,
        'DiastolicBP': dia_bp,
        'BS': bs,
        'BodyTemp': body_temp,
        'HeartRate': heart_rate,
        'RiskLevel': risks
    })
    return df

if __name__ == "__main__":
    df_synthetic = generate_synthetic_data(50)
    print("Membangkitkan data sintetis baru (5 sampel pertama):")
    print(df_synthetic.head())
