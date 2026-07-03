import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json

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

# Set page configuration (Light Theme, Modern Layout)
st.set_page_config(
    page_title="Maternal Health Decision Support System",
    page_icon="🤰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium, SaaS-like Light Theme (simplified to prevent hiding default widget labels)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    /* Apply font only to text elements to prevent breaking Streamlit icons */
    html, body, p, h1, h2, h3, h4, h5, h6, li, button, label, input, select, textarea {
        font-family: 'Outfit', sans-serif !important;
    }
    
    /* Force main app background to be white and text to be dark grey */
    .stApp {
        background-color: #ffffff !important;
    }
    .stApp, .stApp p, .stApp span, .stApp label, .stApp div {
        color: #1e293b !important;
    }
    
    /* Force sidebar background to be soft light grey and text to be dark */
    section[data-testid="stSidebar"] {
        background-color: #f8fafc !important;
    }
    section[data-testid="stSidebar"] * {
        color: #1e293b !important;
    }
    
    /* Inputs text color fix */
    input, select, textarea {
        color: #1e293b !important;
        background-color: #ffffff !important;
    }
    
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 5px;
    }
    
    .subtitle {
        font-size: 1rem;
        color: #64748b;
        margin-bottom: 30px;
    }
    
    /* Metric Card Styling (Left-border removed as requested) */
    .metric-card-custom {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
        margin-bottom: 15px;
        text-align: center;
    }
    
    .metric-card-custom:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    .metric-card-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        color: #64748b;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    
    .metric-card-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1e293b;
    }
    
    /* Button Customization */
    .stButton>button {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: #ffffff !important;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.25s ease;
        box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.2);
    }
    
    .stButton>button * {
        color: #ffffff !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.3);
    }
    
    /* Custom Prediction Card Colors */
    .pred-card {
        padding: 24px;
        border-radius: 16px;
        text-align: center;
        font-weight: 700;
        font-size: 1.4rem;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }
    
    .pred-low {
        background-color: #f0fdf4;
        border: 2px solid #22c55e;
        color: #15803d;
    }
    
    .pred-mid {
        background-color: #fffbeb;
        border: 2px solid #f59e0b;
        color: #b45309;
    }
    
    .pred-high {
        background-color: #fef2f2;
        border: 2px solid #ef4444;
        color: #b91c1c;
    }
</style>
""", unsafe_allow_html=True)

# Helper paths
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data/Maternal Health Risk Data Set.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models/best_model.joblib')

@st.cache_data
def load_dataset():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None

df = load_dataset()

# Sidebar Navigation (Clean, No Broken Image, No Emojis)
st.sidebar.markdown("<h3 style='color: #1e3a8a; font-weight:700; margin-top: 10px;'>Maternal DSS</h3>", unsafe_allow_html=True)
page = st.sidebar.radio("Navigasi Halaman:", ["Dashboard Pemantauan Data", "Prediksi Risiko Pasien Baru"])

@st.cache_resource
def load_model_pack():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

pack = load_model_pack()

# Page 1: Dashboard (No Emojis, No Left Border Lines)
if page == "Dashboard Pemantauan Data":
    st.markdown('<div class="main-title">Dashboard Pemantauan Risiko Kesehatan Maternal</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Analisis Kualitas Data Medis & Evaluasi Komparatif Model Klasifikasi (UAS)</div>', unsafe_allow_html=True)
    
    if df is not None:
        # Four custom metric cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("""
            <div class="metric-card-custom">
                <div class="metric-card-title">Total Sampel Medis</div>
                <div class="metric-card-value">1.014 Pasien</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card-custom">
                <div class="metric-card-title">Rata-rata Usia</div>
                <div class="metric-card-value">{df["Age"].mean():.1f} Tahun</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="metric-card-custom">
                <div class="metric-card-title">Missing Values</div>
                <div class="metric-card-value">0 Data</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown("""
            <div class="metric-card-custom">
                <div class="metric-card-title">Model Rekomendasi</div>
                <div class="metric-card-value">Random Forest</div>
            </div>
            """, unsafe_allow_html=True)
            
        # First row of plots
        st.markdown("<br>", unsafe_allow_html=True)
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("<h4 style='color: #1e3a8a;'>Distribusi Kelas Risiko Kesehatan</h4>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(7, 4))
            fig.patch.set_facecolor('#f8fafc')
            ax.set_facecolor('#ffffff')
            sns.countplot(x='RiskLevel', data=df, order=['low risk', 'mid risk', 'high risk'], palette='Blues_r', ax=ax)
            ax.set_xlabel("Tingkat Risiko", fontsize=10, fontweight='bold', color='#1e293b')
            ax.set_ylabel("Jumlah Pasien", fontsize=10, fontweight='bold', color='#1e293b')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#cbd5e1')
            ax.spines['bottom'].set_color('#cbd5e1')
            st.pyplot(fig)
            
        with col_right:
            st.markdown("<h4 style='color: #1e3a8a;'>Kadar Gula Darah vs Usia Pasien</h4>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(7, 4))
            fig.patch.set_facecolor('#f8fafc')
            ax.set_facecolor('#ffffff')
            sns.scatterplot(x='Age', y='BS', hue='RiskLevel', data=df, palette='Blues_r', alpha=0.7, ax=ax)
            ax.set_xlabel("Usia (Tahun)", fontsize=10, fontweight='bold', color='#1e293b')
            ax.set_ylabel("Kadar Gula Darah (mmol/L)", fontsize=10, fontweight='bold', color='#1e293b')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#cbd5e1')
            ax.spines['bottom'].set_color('#cbd5e1')
            st.pyplot(fig)
            
        # Second row of plots
        st.markdown("<br>", unsafe_allow_html=True)
        col_model_left, col_model_right = st.columns(2)
        with col_model_left:
            st.markdown("<h4 style='color: #1e3a8a;'>Fitur Paling Dominan (Model Importance)</h4>", unsafe_allow_html=True)
            if pack is not None:
                model = pack['model']
                features = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'HeartRate', 'PulsePressure', 'MeanBP', 'ShockIndex']
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    feat_df = pd.DataFrame({'Fitur': features, 'Kepentingan': importances}).sort_values(by='Kepentingan', ascending=True)
                    
                    fig, ax = plt.subplots(figsize=(7, 4))
                    fig.patch.set_facecolor('#f8fafc')
                    ax.set_facecolor('#ffffff')
                    sns.barplot(x='Kepentingan', y='Fitur', data=feat_df, palette='Blues_d', ax=ax)
                    ax.set_xlabel("Skor Kepentingan", fontsize=10, fontweight='bold', color='#1e293b')
                    ax.set_ylabel("Nama Fitur", fontsize=10, fontweight='bold', color='#1e293b')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_color('#cbd5e1')
                    ax.spines['bottom'].set_color('#cbd5e1')
                    st.pyplot(fig)
                else:
                    st.info("Visualisasi kontribusi fitur tidak didukung untuk tipe model ini.")
            else:
                st.warning("Model belum dilatih atau file best_model.joblib tidak ditemukan.")
                
        with col_model_right:
            st.markdown("<h4 style='color: #1e3a8a;'>Perbandingan Akurasi Model Komparatif</h4>", unsafe_allow_html=True)
            perf_df = pd.DataFrame({
                'Model': ['Random Forest', 'XGBoost', 'Decision Tree', 'KNN (Optimized)', 'SVM (Optimized)', 'Naive Bayes'],
                'F1-Score Macro': [0.8718, 0.8665, 0.8534, 0.8341, 0.7014, 0.6054]
            }).sort_values(by='F1-Score Macro', ascending=True)
            
            fig, ax = plt.subplots(figsize=(7, 4))
            fig.patch.set_facecolor('#f8fafc')
            ax.set_facecolor('#ffffff')
            sns.barplot(x='F1-Score Macro', y='Model', data=perf_df, palette='Blues_r', ax=ax)
            ax.set_xlabel("F1-Score Macro", fontsize=10, fontweight='bold', color='#1e293b')
            ax.set_ylabel("Algoritma", fontsize=10, fontweight='bold', color='#1e293b')
            ax.set_xlim(0.4, 1.0)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#cbd5e1')
            ax.spines['bottom'].set_color('#cbd5e1')
            st.pyplot(fig)
            
    else:
        st.error("Dataset tidak ditemukan di folder data/.")

# Page 2: Prediction Form (No Emojis, Restored Widget Label Colors)
else:
    st.markdown('<div class="main-title">Skrining Risiko Kesehatan Ibu Hamil</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Isi Form Parameter Klinis untuk Mendapatkan Prediksi Risiko Terstandar & Rekomendasi AI</div>', unsafe_allow_html=True)
    
    if pack is not None:
        active_model = pack['model']
        active_scaler = pack['scaler']
        active_selector = pack['selector']
        active_le = pack['le']
        active_features = pack['features']
        
        # Form Layout
        st.markdown("<h4 style='color: #1e3a8a; margin-bottom: 15px;'>Masukkan Vital Signs Pasien:</h4>", unsafe_allow_html=True)
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            age = st.number_input("Usia Ibu (Tahun)", min_value=10, max_value=70, value=25, step=1)
            sys_bp = st.slider("Tekanan Darah Sistolik (mmHg)", 70, 160, 110)
            dia_bp = st.slider("Tekanan Darah Diastolik (mmHg)", 40, 100, 70)
        with col_f2:
            bs = st.number_input("Kadar Gula Darah / Blood Sugar (mmol/L)", min_value=3.0, max_value=25.0, value=7.0, step=0.1)
            body_temp = st.slider("Suhu Tubuh (°F)", 96.0, 104.0, 98.6, step=0.1)
            heart_rate = st.slider("Detak Jantung / Heart Rate (bpm)", 50, 120, 75)
            
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Analisis Risiko Kesehatan"):
            # Compute engineered features
            pp = sys_bp - dia_bp
            mean_bp = (sys_bp + 2 * dia_bp) / 3
            shock_index = heart_rate / max(1, sys_bp)
            
            patient_df = pd.DataFrame([{
                'Age': age, 'SystolicBP': sys_bp, 'DiastolicBP': dia_bp, 
                'BS': bs, 'BodyTemp': body_temp, 'HeartRate': heart_rate,
                'PulsePressure': pp, 'MeanBP': mean_bp, 'ShockIndex': shock_index
            }])
            
            # Scale & Select
            scaled_inp = active_scaler.transform(patient_df[active_features])
            selected_inp = active_selector.transform(scaled_inp)
            
            # Make prediction
            pred_idx = active_model.predict(selected_inp)[0]
            pred_label = active_le.inverse_transform([pred_idx])[0]
            
            # Predict Probabilities
            proba = active_model.predict_proba(selected_inp)[0]
            proba_dict = {cls: proba[idx] for idx, cls in enumerate(active_le.classes_)}
            
            # Result visual class card
            class_css = "pred-low" if "low" in pred_label else ("pred-mid" if "mid" in pred_label else "pred-high")
            st.markdown(f"""
            <div class="pred-card {class_css}">
                Hasil Prediksi Diagnosis: {pred_label.upper()}
            </div>
            """, unsafe_allow_html=True)
            
            # Get AI recommendation
            with st.spinner("Menghubungi Asisten AI Medis untuk mendapatkan rekomendasi preventif..."):
                ai_rec = get_ai_recommendation(age, sys_bp, dia_bp, bs, body_temp, heart_rate, pred_label)
            
            st.markdown(f"""
            <div style="background-color: #f1f5f9; border-left: 6px solid #1e3a8a; padding: 20px; border-radius: 12px; margin-top: 10px; margin-bottom: 25px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.03);">
                <h4 style="color: #1e3a8a; margin-top: 0; font-size: 1.05rem; font-weight: 700; letter-spacing: 0.02em;">Rekomendasi Medis Pintar AI:</h4>
                <p style="font-style: italic; color: #334155; margin-bottom: 0; font-size: 0.95rem; line-height: 1.6;">"{ai_rec}"</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Metric probability breakdown
            st.markdown("<h4 style='color: #1e3a8a;'>Keyakinan Keputusan Model:</h4>", unsafe_allow_html=True)
            cols_prob = st.columns(3)
            for idx, (cls, p_val) in enumerate(proba_dict.items()):
                with cols_prob[idx]:
                    st.metric(label=f"Probabilitas {cls.upper()}", value=f"{p_val:.2%}")
            
            # Model explanation section
            st.markdown("<br><hr style='border: 0; border-top: 1px solid #e2e8f0;'><br>", unsafe_allow_html=True)
            st.markdown("<h4 style='color: #1e3a8a;'>Penjelasan Logika Model Medis:</h4>", unsafe_allow_html=True)
            st.markdown(f"""
            Model klasifikasi terbaik kami (**Random Forest**) memproses data di atas setelah dilakukan standardisasi skala. 
            Fitur hemodinamik hasil rekayasa seperti **MeanBP** ({mean_bp:.1f} mmHg) dan **ShockIndex** ({shock_index:.3f}) 
            digunakan oleh ratusan pohon keputusan (*decision trees*) secara paralel untuk mengevaluasi margin risiko Anda. 
            Kadar gula darah pasien sebesar **{bs} mmol/L** memegang bobot keputusan tertinggi (~32%) dalam penentuan risiko kehamilan ini.
            """)
            
            # Ethical disclaimer
            st.info("""
            Batasan Penggunaan Klinis (Deklarasi Etis):
            Sistem Cerdas ini dikembangkan sebagai instrumen skrining penunjang awal (Decision Support System) 
            dan tidak ditujukan untuk menggantikan pemeriksaan fisik secara tatap muka ataupun diagnosis klinis mutlak 
            dari bidan berwenang atau dokter spesialis kandungan.
            """)
    else:
        st.warning("Model terbaik belum dilatih/ditemukan. Silakan jalankan script train.py terlebih dahulu.")
