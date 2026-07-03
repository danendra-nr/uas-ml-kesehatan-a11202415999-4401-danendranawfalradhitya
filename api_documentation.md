# Dokumentasi API - Layanan Prediksi Risiko Kesehatan Maternal

Layanan RESTful API ini dibangun menggunakan **FastAPI** untuk memfasilitasi prediksi tingkat risiko kesehatan kehamilan maternal secara real-time berdasarkan data klinis pasien, lengkap dengan rekomendasi preventif pintar berbasis AI.

---

## Basis URL (Base URL)
*   **Lokal (Development):** `http://127.0.0.1:8000`
*   **Produksi (Production Mockup):** `https://your-url.com`

---

## Endpoint Layanan

### 1. Cek Status Layanan (Health Check)
Mengembalikan status aktif dari API.

*   **URL:** `/`
*   **Metode:** `GET`
*   **Headers:** Tidak ada
*   **Respon Sukses (200 OK):**
    ```json
    {
      "message": "Welcome to the Maternal Health Risk Prediction API"
    }
    ```

---

### 2. Prediksi Risiko Maternal & Rekomendasi AI
Menerima parameter klinis pasien, menghitung rekayasa fitur hemodinamik, memprediksi tingkat risiko kesehatan (Low, Mid, High Risk), dan menghasilkan rekomendasi medis preventif dari AI.

*   **URL:** `/predict`
*   **Metode:** `POST`
*   **Headers:** 
    *   `Content-Type: application/json`
*   **Format Request Body (JSON):**
    | Nama Variabel | Tipe Data | Deskripsi Klinis | Range Contoh |
    | :--- | :--- | :--- | :--- |
    | `Age` | Integer | Usia Ibu (Tahun) | `25` |
    | `SystolicBP` | Integer | Tekanan Darah Sistolik (mmHg) | `120` |
    | `DiastolicBP` | Integer | Tekanan Darah Diastolik (mmHg) | `80` |
    | `BS` | Float | Kadar Gula Darah (mmol/L) | `7.5` |
    | `BodyTemp` | Float | Suhu Tubuh (°F) | `98.0` |
    | `HeartRate` | Integer | Detak Jantung (bpm) | `70` |

    *Contoh Request Body:*
    ```json
    {
      "Age": 29,
      "SystolicBP": 130,
      "DiastolicBP": 85,
      "BS": 7.8,
      "BodyTemp": 98.6,
      "HeartRate": 75
    }
    ```

*   **Format Response Body (JSON - 200 OK):**
    ```json
    {
      "prediction": "mid risk",
      "probabilities": {
        "low risk": 0.125,
        "mid risk": 0.725,
        "high risk": 0.150
      },
      "ai_recommendation": "Kondisi kehamilan Anda menunjukkan tingkat risiko sedang. Disarankan untuk membatasi konsumsi gula berlebih, memantau tekanan darah secara berkala, dan segera menjadwalkan pemeriksaan rutin dengan bidan atau dokter kandungan spesialis.",
      "features": {
        "PulsePressure": 45,
        "MeanBP": 100.0,
        "ShockIndex": 0.5769
      }
    }
    ```

*   **Respon Eror (500 Internal Server Error):**
    Terjadi jika model machine learning (`best_model.joblib`) belum dilatih.
    ```json
    {
      "detail": "Model file not found. Run training first."
    }
    ```

---

## Contoh Integrasi Klien (Python requests)

Anda dapat mengintegrasikan API ini ke aplikasi mobile atau web menggunakan script Python berikut:

```python
import requests

url = "http://127.0.0.1:8000/predict"
data = {
    "Age": 30,
    "SystolicBP": 140,
    "DiastolicBP": 90,
    "BS": 12.0,
    "BodyTemp": 98.0,
    "HeartRate": 80
}

response = requests.post(url, json=data)
if response.status_code == 200:
    res = response.json()
    print("Diagnosis:", res['prediction'].upper())
    print("Rekomendasi AI:", res['ai_recommendation'])
else:
    print("Eror:", response.text)
```
