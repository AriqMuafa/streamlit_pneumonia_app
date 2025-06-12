import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Deteksi Pneumonia AI",
    page_icon="🫁",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Fungsi untuk Memuat Model ---
@st.cache_resource
def load_models():
    """Memuat semua model dari direktori /models dan menyimpannya di cache."""
    models_dir = 'models'
    model_paths = {
        'Model Utama (ResNet50V2)': os.path.join(models_dir, 'model_utama.h5'),
        'CNN Kustom': os.path.join(models_dir, 'model_cnn.h5'),
        'VGG16': os.path.join(models_dir, 'model_vgg16.h5')
    }
    
    loaded_models = {}
    for name, path in model_paths.items():
        if os.path.exists(path):
            try:
                # --- PERUBAHAN DI SINI ---
                # Tambahkan compile=False untuk mengabaikan loading optimizer
                loaded_models[name] = tf.keras.models.load_model(path, compile=False)
            except Exception as e:
                st.error(f"Gagal memuat model {name}: {e}")
                loaded_models[name] = None 
        else:
            st.warning(f"File model tidak ditemukan untuk {name} di path: {path}")
            loaded_models[name] = None
            
    return loaded_models

# --- Fungsi Pra-pemrosesan Gambar ---
def preprocess_image(image, target_size=(224, 224)):
    """Mengubah ukuran, format, dan normalisasi gambar."""
    img = image.convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# --- Memuat Model ---
models = load_models()
available_models = [name for name, model in models.items() if model is not None]

# --- Tampilan Antarmuka (UI) dengan Streamlit ---

st.title("🫁 Deteksi Pneumonia Berbasis AI")
st.markdown("Unggah gambar sinar-X dada untuk mendapatkan prediksi pneumonia menggunakan model *Deep Learning*.")

with st.sidebar:
    st.header("Pengaturan")
    if not available_models:
        st.error("Tidak ada model yang berhasil dimuat. Periksa file model Anda.")
        selected_model_name = None
    else:
        selected_model_name = st.selectbox(
            "Pilih Model Analisis:",
            options=available_models,
            index=0
        )

uploaded_file = st.file_uploader(
    "Pilih atau seret gambar sinar-X ke sini",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None and selected_model_name is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Gambar yang Diunggah", use_column_width=True)

    with st.spinner(f"Menganalisis menggunakan {selected_model_name}..."):
        processed_image = preprocess_image(image)
        
        model = models[selected_model_name]
        prediction = model.predict(processed_image)
        
        is_pneumonia = prediction[0][0] > 0.5
        
        with col2:
            st.subheader("Hasil Prediksi")
            if is_pneumonia:
                st.error(f"**Status: PNEUMONIA**")
                st.metric(label="Tingkat Keyakinan Model", value=f"{prediction[0][0]*100:.2f}%")
            else:
                st.success(f"**Status: NORMAL**")
                st.metric(label="Tingkat Keyakinan Model", value=f"{(1-prediction[0][0])*100:.2f}%")
            
            st.info(f"Model yang digunakan: **{selected_model_name}**")
            
    st.markdown("---")
    st.markdown("""
    ***Disclaimer:*** *Hasil prediksi ini dihasilkan oleh model kecerdasan buatan dan **tidak boleh** digunakan sebagai pengganti diagnosis medis profesional. Selalu konsultasikan dengan dokter atau ahli radiologi.*
    """)
else:
    st.info("Silakan unggah gambar untuk memulai analisis.")
