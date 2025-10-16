# app_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Sistem Pakar Penentuan UKT", layout="centered")

st.title("Sistem Pakar Penentuan UKT (Demo)")
st.write("Aplikasi demo: masukkan profil mahasiswa -> prediksi kategori UKT menggunakan model RF.")

# --- Load model ---
@st.cache_resource
def load_model(path="model_pipeline.joblib"):
    obj = joblib.load(path)
    return obj["pipeline"], obj["label_encoder"]

try:
    pipeline, label_encoder = load_model()
    model_ready = True
except Exception as e:
    st.error(f"Gagal memuat model: {e}. Pastikan model_pipeline.joblib berada di folder yang sama.")
    model_ready = False

# --- Sidebar: input form ---
st.sidebar.header("Input data mahasiswa")
def user_input_form():
    pendapatan = st.sidebar.number_input("Pendapatan Orang Tua (Rp/bulan)", min_value=0, value=3000000, step=100000)
    tanggungan = st.sidebar.slider("Tanggungan Keluarga (orang)", 1, 10, 3)
    pekerjaan = st.sidebar.selectbox("Pekerjaan Orang Tua", ["PNS", "Swasta", "Petani", "Wiraswasta", "Tidak Bekerja"])
    kepemilikan = st.sidebar.selectbox("Kepemilikan Rumah", ["Milik Sendiri", "Kontrak", "Rumah Dinas", "Menumpang"])
    tagihan = st.sidebar.number_input("Tagihan Listrik (Rp/bulan)", min_value=0, value=150000, step=10000)
    kendaraan = st.sidebar.selectbox("Kendaraan", ["Tidak Ada", "Motor", "Mobil", "Motor+Mobil"])
    ipk = st.sidebar.slider("Nilai IPK", 0.0, 4.0, 3.0, step=0.01)
    beasiswa = st.sidebar.selectbox("Beasiswa", ["Ya", "Tidak"])
    return {
        "Pendapatan_Ortu": pendapatan,
        "Tanggungan_Keluarga": tanggungan,
        "Pekerjaan_Ortu": pekerjaan,
        "Kepemilikan_Rumah": kepemilikan,
        "Tagihan_Listrik": tagihan,
        "Kendaraan": kendaraan,
        "Nilai_IPK": round(ipk, 2),
        "Beasiswa": beasiswa
    }

input_data = user_input_form()

st.subheader("Profil Mahasiswa (preview)")
st.json(input_data)

# --- Predict ---
if model_ready:
    if st.button("Prediksi Kategori UKT"):
        df_input = pd.DataFrame([input_data])
        pred = pipeline.predict(df_input)[0]
        proba = pipeline.predict_proba(df_input)[0]  # probabilities for each class
        # decode label
        pred_label = label_encoder.inverse_transform([pred])[0]
        st.success(f"Prediksi: {pred_label}")
        st.write("Probabilitas untuk setiap kelas:")
        proba_df = pd.DataFrame({
            "Kategori_UKT": label_encoder.classes_,
            "Probabilitas": proba
        }).sort_values("Probabilitas", ascending=False).reset_index(drop=True)
        st.table(proba_df)

        st.markdown("**Interpretasi singkat:**")
        st.write("- Prediksi menunjukkan kelas UKT yang paling mungkin berdasarkan profile input.")
        st.write("- Jika probabilitas terdistribusi tipis (mirip), pertimbangkan verifikasi manual.")
        st.write("- Model ini berbasis data dummy; untuk produksi perlu validasi dengan data sesungguhnya.")

# --- Opsi: lihat data training (jika ada) ---
if st.checkbox("Tampilkan contoh dataset (5 baris)"):
    try:
        df_raw = pd.read_csv("dummy_data_ukt_1000.csv")
        st.dataframe(df_raw.head())
    except Exception as e:
        st.error(f"Gagal membaca CSV: {e}")

# --- Footer / info ---
st.markdown("---")
st.caption("Catatan: model dan data bersifat demo. Untuk penelitian: lakukan validasi, cross-validation, tuning hyperparameter, fairness check, dan uji sensivitas.")
