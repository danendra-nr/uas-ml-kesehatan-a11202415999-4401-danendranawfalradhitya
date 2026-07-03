# Data Dictionary - Maternal Health Risk Dataset

Dokumen ini berisi informasi kamus data (*data dictionary*) untuk dataset tingkat risiko kesehatan ibu hamil (*Maternal Health Risk*).

## Fitur-Fitur Prediktor Medis Primer (Original Features)

| Nama Fitur | Tipe Data | Deskripsi | Rentang Nilai |
|---|---|---|---|
| **Age** | Integer | Usia pasien/ibu hamil saat pemeriksaan kesehatan (Tahun). | 10 - 70 |
| **SystolicBP** | Integer | Tekanan darah sistolik, yaitu batas atas tekanan arteri saat jantung berkontraksi memompa darah (mmHg). | 70 - 160 |
| **DiastolicBP** | Integer | Tekanan darah diastolik, yaitu batas bawah tekanan arteri saat jantung beristirahat di sela-sela denyutan (mmHg). | 49 - 100 |
| **BS** | Float | Kadar gula darah (*Blood Sugar*) pasien setelah puasa atau acak (mmol/L). | 6.0 - 19.0 |
| **BodyTemp** | Float | Suhu tubuh pasien saat diukur menggunakan termometer (°F). | 98.0 - 103.0 |
| **HeartRate** | Integer | Frekuensi denyut jantung istirahat pasien per menit (bpm). | 7 - 90 |

---

## Fitur Hasil Rekayasa (Engineered Features)

| Nama Fitur | Tipe Data | Rumus / Perhitungan | Deskripsi |
|---|---|---|---|
| **PulsePressure** | Integer | `SystolicBP - DiastolicBP` | Selisih antara tekanan darah sistolik dan diastolik. Secara klinis menggambarkan kelenturan pembuluh darah arteri. |
| **MeanBP** | Float | `(SystolicBP + 2 * DiastolicBP) / 3` | *Mean Arterial Pressure* (MAP), yaitu rata-rata tekanan arteri selama satu siklus jantung penuh. Berguna mengukur perfusi organ tubuh. |
| **ShockIndex** | Float | `HeartRate / SystolicBP` | Indeks keparahan syok hemodinamik. Digunakan secara klinis untuk menilai potensi ketidakstabilan sirkulasi darah. |

---

## Label Target (Target Variable)

| Nama Kolom | Tipe Data | Kategori Kelas | Deskripsi |
|---|---|---|---|
| **RiskLevel** | String | `low risk`, `mid risk`, `high risk` | Klasifikasi tingkat risiko kesehatan ibu hamil berdasarkan diagnosis klinis medis. |
| **RiskLevel_Encoded** | Integer | `0` (low risk), `1` (mid risk), `2` (high risk) | Representasi numerik hasil Label Encoding target untuk input komputasi machine learning. |
