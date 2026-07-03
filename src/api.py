import os
import logging
import joblib
import pandas as pd
import urllib.request
import json
from datetime import datetime
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def load_dotenv(dotenv_path=".env"):
    if os.path.exists(dotenv_path):
        logger.info(f"Memuat variabel lingkungan dari {dotenv_path}")
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    val_str = value.strip().strip("'").strip('"')
                    os.environ[key.strip()] = val_str

load_dotenv()

app = FastAPI(title="Maternal Health Risk API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/best_model.joblib')
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file tidak ditemukan: {MODEL_PATH}")

data     = joblib.load(MODEL_PATH)
model    = data['model']
scaler   = data['scaler']
le       = data['le']
features = data['features']
_api_start_time = datetime.now()


class PatientData(BaseModel):
    Age:         int   = Field(..., ge=10, le=70)
    SystolicBP:  int   = Field(..., ge=70, le=200)
    DiastolicBP: int   = Field(..., ge=40, le=130)
    BS:          float = Field(..., ge=3.0, le=30.0)
    BodyTemp:    float = Field(..., ge=95.0, le=108.0)
    HeartRate:   int   = Field(..., ge=40, le=200)

    @validator('DiastolicBP')
    def diastolic_less_than_systolic(cls, v, values):
        if 'SystolicBP' in values and v >= values['SystolicBP']:
            raise ValueError("DiastolicBP harus lebih kecil dari SystolicBP")
        return v


def _build_input_df(patient_dict: dict) -> pd.DataFrame:
    df = pd.DataFrame([patient_dict])
    df['PulsePressure'] = df['SystolicBP'] - df['DiastolicBP']
    if 'MeanBP' in features:
        df['MeanBP'] = (df['SystolicBP'] + 2 * df['DiastolicBP']) / 3
    if 'ShockIndex' in features:
        df['ShockIndex'] = df['HeartRate'] / df['SystolicBP'].replace(0, 1)
    return df[features]


def _predict_with_confidence(input_df: pd.DataFrame):
    scaled   = scaler.transform(input_df)
    pred_enc = model.predict(scaled)
    label    = le.inverse_transform(pred_enc)[0]
    confidence = {}
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(scaled)[0]
        confidence = {cls: round(float(p), 4) for cls, p in zip(le.classes_, proba)}
    return label, confidence


def _generate_ai_recommendation(patient_data: dict, prediction: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model_name = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    
    if not api_key or api_key == "your_api_key_here":
        logger.info("OpenAI API key is missing or placeholder. Skipping AI recommendation.")
        return ""
        
    prompt = (
        f"Sebagai asisten medis AI obstetrician pintar, berikan analisis ringkas dan rekomendasi klinis "
        f"untuk pasien ibu hamil berikut:\n"
        f"- Usia: {patient_data['Age']} tahun\n"
        f"- Tekanan Darah: {patient_data['SystolicBP']}/{patient_data['DiastolicBP']} mmHg\n"
        f"- Gula Darah: {patient_data['BS']} mmol/L\n"
        f"- Suhu Tubuh: {patient_data['BodyTemp']} °F\n"
        f"- Detak Jantung: {patient_data['HeartRate']} bpm\n"
        f"- Hasil Prediksi Tingkat Risiko: {prediction.upper()}\n\n"
        f"Berikan jawaban dalam Bahasa Indonesia yang ramah, profesional, padat (maksimal 3-4 kalimat), "
        f"dan sertakan langkah preventif medis konkret."
    )
    
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system", 
                "content": "Anda adalah dokter spesialis kandungan (obstetrician) AI yang ahli, memberikan saran medis ringkas, ramah, dan profesional."
            },
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }
    
    try:
        url = f"{base_url.rstrip('/')}/chat/completions"
        logger.info(f"Menghubungi OpenAI Compatible API di: {url} (Model: {model_name})")
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode('utf-8'),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            },
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=8.0) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            ai_text = res_data['choices'][0]['message']['content'].strip()
            logger.info("Rekomendasi AI berhasil didapatkan.")
            return ai_text
    except Exception as e:
        logger.error(f"Gagal memanggil API OpenAI compatible: {e}")
        return f"Rekomendasi AI otomatis saat ini tidak tersedia (karena kendala koneksi API)."


@app.get("/")
def read_root():
    return {"message": "Maternal Health Risk Prediction API", "uptime_since": _api_start_time.isoformat()}


@app.get("/health")
def health_check():
    api_key = os.getenv("OPENAI_API_KEY")
    ai_status = "active" if (api_key and api_key != "your_api_key_here") else "disabled"
    return {
        "status": "ok",
        "model_type": type(model).__name__,
        "features_used": features,
        "label_classes": le.classes_.tolist(),
        "api_version": "2.0.0",
        "openai_integration": ai_status,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict")
def predict(patient: PatientData):
    logger.info("Prediksi tunggal: %s", patient.dict())
    try:
        input_df = _build_input_df(patient.dict())
        label, confidence = _predict_with_confidence(input_df)
        ai_recommendation = _generate_ai_recommendation(patient.dict(), label)
        return {
            "prediction": label, 
            "confidence": confidence, 
            "ai_recommendation": ai_recommendation,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def predict_batch(patients: List[PatientData]):
    if len(patients) > 100:
        raise HTTPException(status_code=400, detail="Maksimal 100 pasien per request batch.")
    logger.info("Prediksi batch: %d pasien", len(patients))
    results = []
    for i, p in enumerate(patients):
        try:
            input_df = _build_input_df(p.dict())
            label, confidence = _predict_with_confidence(input_df)
            results.append({"index": i, "prediction": label, "confidence": confidence, "status": "success"})
        except Exception as e:
            results.append({"index": i, "status": "error", "detail": str(e)})
    return {"total": len(patients), "results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
