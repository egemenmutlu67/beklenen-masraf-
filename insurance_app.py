import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Evin Masrafını Hesap Etme", layout="centered")

st.title("🏠 Evin Masrafını Hesap Etme")
st.write("Aşağıdaki bilgileri doldurarak evin sigorta masrafını tahmin edebilirsiniz.")

# Kullanıcıdan giriş al
ev_durumu = st.selectbox("Ev Durumu", ["Ev Sahibi", "Kiralık"])
evcil_hayvan = st.selectbox("Evcil Hayvan Sahibi misiniz?", ["yes", "no"])
bolge = st.selectbox("Bölge", [
    "Batı Bölgesi", 
    "İç Anadolu ve Karadeniz Bölgesi", 
    "Akdeniz ve Ege İç Kesimleri Bölgesi", 
    "Doğu ve Güneydoğu Anadolu Bölgesi"
])

# Evin yaşı için eğitim verisine uygun kategoriler
evin_yasi_kategori = st.selectbox("Evin Yaşı", [
    "0-9", "10-19", "20-29", "30-39", 
    "40-49", "50-59", "60-69", "70-79", 
    "80-89", "90-99", "100+"
])

# Çocuk sayısı için eğitim verisine uygun kategoriler
cocuk_sayisi_kategori = st.selectbox("Çocuk Sayısı", ["Yüksek", "Normal", "Düşük"])

# Kozmetik durum için seçenekler
kozmetik_durum = st.selectbox("Evin Kozmetik Durumu", ["İyi", "Normal", "Kötü"])

# Girişleri uygun formatta DataFrame’e dönüştür
input_dict = {
    "Ev Durumu": ev_durumu,
    "Evcil Hayvan Sahibi": 1 if evcil_hayvan == "yes" else 0,
    "Bolge": bolge,
    "Katagorik_Evin_Yasi": evin_yasi_kategori,
    "Katagorik_Cocuk_Sayisi": cocuk_sayisi_kategori,
    "Katagorik_Evin_Kozmetik_Durumu": kozmetik_durum
}

input_df = pd.DataFrame([input_dict])

# Model ve kolonları yükle
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model_columns.pkl", "rb") as f:
        model_columns = pickle.load(f)

    # one-hot encoding uygula (özellikleri modele uygun hale getir)
    input_processed = pd.get_dummies(input_df)

    # Eksik kolonları ekle (0 olarak)
    for col in model_columns:
        if col not in input_processed.columns:
            input_processed[col] = 0

    # Kolonları sırala
    input_processed = input_processed[model_columns]

    # Tahmin butonu
    if st.button("Masrafı Hesapla"):
        prediction = model.predict(input_processed)[0]
        st.success(f"💸 Tahmini Sigorta Masrafınız: {prediction:,.2f} ₺")

except FileNotFoundError:
    st.error("Model dosyaları bulunamadı. Lütfen 'model.pkl' ve 'model_columns.pkl' dosyalarının mevcut olduğundan emin olun.")
