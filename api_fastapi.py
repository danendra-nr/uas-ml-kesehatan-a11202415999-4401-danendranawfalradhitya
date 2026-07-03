import os
import json
import joblib
import requests
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Maternal Health Risk Predictor API", version="1.0.0")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models/best_model.joblib')

# Load .env variables manually to avoid extra dependencies
def load_env(env_path='.env'):
    if os.path.exists(env_path):
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, val = line.split('=', 1)
                        os.environ[key.strip()] = val.strip()

load_env()

class PatientInput(BaseModel):
    Age: int
    SystolicBP: int
    DiastolicBP: int
    BS: float
    BodyTemp: float
    HeartRate: int

def get_ai_recommendation(age, sys_bp, dia_bp, bs, body_temp, heart_rate, prediction):
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
    
    if not api_key:
        return "Rekomendasi Pintar AI tidak tersedia: API Key belum dikonfigurasi di file .env."
        
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    prompt = (
        f"Pasien Ibu Hamil berusia {age} tahun dengan kondisi klinis:\n"
        f"- Tekanan Darah Sistolik: {sys_bp} mmHg\n"
        f"- Tekanan Darah Diastolik: {dia_bp} mmHg\n"
        f"- Kadar Gula Darah (BS): {bs} mmol/L\n"
        f"- Suhu Tubuh: {body_temp} F\n"
        f"- Detak Jantung: {heart_rate} bpm\n"
        f"Model Machine Learning mendiagnosis tingkat risiko: {prediction}.\n\n"
        f"Berikan rekomendasi medis preventif singkat dan etis dalam bahasa Indonesia (maksimal 3 kalimat) "
        f"yang sesuai dengan tingkat risiko tersebut. Selalu sertakan saran untuk berkonsultasi dengan bidan atau dokter spesialis kandungan."
    )
    
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Anda adalah asisten AI medis klinis yang memberikan saran etis preventif untuk ibu hamil."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }
    
    try:
        response = requests.post(f"{base_url}/chat/completions", headers=headers, json=data, timeout=8)
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        else:
            return f"Gagal menghubungi asisten AI (Status HTTP: {response.status_code}). Silakan hubungi bidan atau dokter terdekat."
    except Exception as e:
        return f"Koneksi ke asisten AI terputus ({str(e)}). Harap konsultasikan kesehatan Anda ke dokter terdekat."

@app.get("/")
def read_root():
    return {"message": "Welcome to the Maternal Health Risk Prediction API"}

@app.post("/predict")
def predict_risk(patient: PatientInput):
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=500, detail="Model file not found. Run training first.")
        
    pack = joblib.load(MODEL_PATH)
    active_model = pack['model']
    active_scaler = pack['scaler']
    active_selector = pack['selector']
    active_le = pack['le']
    active_features = pack['features']
    
    # Feature engineering
    pp = patient.SystolicBP - patient.DiastolicBP
    mean_bp = (patient.SystolicBP + 2 * patient.DiastolicBP) / 3
    shock_index = patient.HeartRate / max(1, patient.SystolicBP)
    
    patient_df = pd.DataFrame([{
        'Age': patient.Age, 'SystolicBP': patient.SystolicBP, 'DiastolicBP': patient.DiastolicBP, 
        'BS': patient.BS, 'BodyTemp': patient.BodyTemp, 'HeartRate': patient.HeartRate,
        'PulsePressure': pp, 'MeanBP': mean_bp, 'ShockIndex': shock_index
    }])
    
    scaled_inp = active_scaler.transform(patient_df[active_features])
    selected_inp = active_selector.transform(scaled_inp)
    
    pred_idx = active_model.predict(selected_inp)[0]
    pred_label = active_le.inverse_transform([pred_idx])[0]
    
    proba = active_model.predict_proba(selected_inp)[0]
    proba_dict = {cls: float(proba[idx]) for idx, cls in enumerate(active_le.classes_)}
    
    # Get AI recommendation based on variables
    ai_rec = get_ai_recommendation(
        patient.Age, patient.SystolicBP, patient.DiastolicBP, 
        patient.BS, patient.BodyTemp, patient.HeartRate, pred_label
    )
    
    return {
        "prediction": pred_label,
        "probabilities": proba_dict,
        "ai_recommendation": ai_rec,
        "features": {
            "PulsePressure": pp,
            "MeanBP": mean_bp,
            "ShockIndex": shock_index
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
