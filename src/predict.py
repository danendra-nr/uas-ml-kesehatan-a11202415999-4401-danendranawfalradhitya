import os
import sys
import joblib
import pandas as pd

def predict_patient(age, systolic_bp, diastolic_bp, bs, body_temp, heart_rate):
    model_path = os.path.join(os.path.dirname(__file__), '../models/best_model.joblib')
    if not os.path.exists(model_path):
        print("[ERROR] File model best_model.joblib tidak ditemukan! Jalankan train.py terlebih dahulu.")
        sys.exit(1)
        
    best_pack = joblib.load(model_path)
    active_model = best_pack['model']
    active_scaler = best_pack['scaler']
    active_selector = best_pack['selector']
    active_le = best_pack['le']
    active_features = best_pack['features']
    
    # Preprocess input patient data
    patient_data = {
        'Age': age,
        'SystolicBP': systolic_bp,
        'DiastolicBP': diastolic_bp,
        'BS': bs,
        'BodyTemp': body_temp,
        'HeartRate': heart_rate
    }
    
    inp = pd.DataFrame([patient_data])
    inp['PulsePressure'] = inp['SystolicBP'] - inp['DiastolicBP']
    inp['MeanBP']        = (inp['SystolicBP'] + 2 * inp['DiastolicBP']) / 3
    inp['ShockIndex']    = inp['HeartRate'] / inp['SystolicBP'].replace(0, 1)
    
    # Scale & Feature Select
    scaled_inp = active_scaler.transform(inp[active_features])
    selected_inp = active_selector.transform(scaled_inp)
    
    # Predict
    pred_idx = active_model.predict(selected_inp)[0]
    pred_label = active_le.inverse_transform([pred_idx])[0]
    
    proba = active_model.predict_proba(selected_inp)[0]
    proba_dict = {cls: f"{p:.2%}" for cls, p in zip(active_le.classes_, proba)}
    
    return pred_label, proba_dict

if __name__ == "__main__":
    # Example test CLI
    if len(sys.argv) == 7:
        age = float(sys.argv[1])
        sys_bp = float(sys.argv[2])
        dia_bp = float(sys.argv[3])
        bs = float(sys.argv[4])
        temp = float(sys.argv[5])
        hr = float(sys.argv[6])
    else:
        # Default test
        print("Menggunakan data default (Pasien High Risk expected): 30th, 140/90, BS: 13.0, HR: 70")
        age, sys_bp, dia_bp, bs, temp, hr = 30, 140, 90, 13.0, 98.0, 70
        
    label, probas = predict_patient(age, sys_bp, dia_bp, bs, temp, hr)
    print(f"Diagnosis Risiko: {label.upper()}")
    print("Probabilitas Kelas:")
    for cls, val in probas.items():
        print(f"  - {cls.upper()}: {val}")
