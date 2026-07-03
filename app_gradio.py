import os
import joblib
import pandas as pd

try:
    import gradio as gr
except ImportError:
    print("[INFO] Gradio tidak terinstall, gunakan: pip install gradio")

# Path model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models/best_model.joblib')

def predict_maternal_risk(age, systolic_bp, diastolic_bp, bs, body_temp, heart_rate):
    if not os.path.exists(MODEL_PATH):
        return "Model belum dilatih. Silakan jalankan train.py terlebih dahulu."
        
    pack = joblib.load(MODEL_PATH)
    active_model = pack['model']
    active_scaler = pack['scaler']
    active_selector = pack['selector']
    active_le = pack['le']
    active_features = pack['features']
    
    # Feature engineering
    pp = systolic_bp - diastolic_bp
    mean_bp = (systolic_bp + 2 * diastolic_bp) / 3
    shock_index = heart_rate / max(1, systolic_bp)
    
    patient_df = pd.DataFrame([{
        'Age': age, 'SystolicBP': systolic_bp, 'DiastolicBP': diastolic_bp, 
        'BS': bs, 'BodyTemp': body_temp, 'HeartRate': heart_rate,
        'PulsePressure': pp, 'MeanBP': mean_bp, 'ShockIndex': shock_index
    }])
    
    scaled_inp = active_scaler.transform(patient_df[active_features])
    selected_inp = active_selector.transform(scaled_inp)
    
    pred_idx = active_model.predict(selected_inp)[0]
    pred_label = active_le.inverse_transform([pred_idx])[0]
    
    proba = active_model.predict_proba(selected_inp)[0]
    proba_str = ", ".join(f"{cls.upper()}: {proba[idx]:.2%}" for idx, cls in enumerate(active_le.classes_))
    
    return f"Hasil Diagnosis: {pred_label.upper()}\nKeyakinan Model: {proba_str}"

# Interface design (no emojis)
if 'gr' in globals():
    demo = gr.Interface(
        fn=predict_maternal_risk,
        inputs=[
            gr.Slider(10, 70, value=25, label="Usia Pasien"),
            gr.Slider(70, 160, value=110, label="Sistolik (mmHg)"),
            gr.Slider(40, 100, value=70, label="Diastolik (mmHg)"),
            gr.Slider(5.0, 20.0, value=7.0, label="Gula Darah (mmol/L)"),
            gr.Slider(98.0, 104.0, value=98.6, label="Suhu Tubuh (°F)"),
            gr.Slider(50, 120, value=75, label="Detak Jantung (bpm)")
        ],
        outputs="text",
        title="Maternal Health Risk Classifier (Gradio UI)",
        description="Aplikasi penunjang keputusan skrining risiko klinis ibu hamil menggunakan Random Forest.",
        article="*Peringatan Etis: Aplikasi ini hanya bertindak sebagai CDSS pendukung awal, bukan diagnosis final medis.*"
    )

    if __name__ == "__main__":
        demo.launch(server_name="127.0.0.1", server_port=7860)
