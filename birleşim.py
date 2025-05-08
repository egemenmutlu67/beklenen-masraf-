import streamlit as st
import pandas as pd
import pickle
import os

# Sayfa ayarları
st.set_page_config(page_title="Evin Masrafını Hesap Etme", layout="centered")
st.title("🏠 Evin Masrafını Hesap Etme")
st.write("Aşağıdaki bilgileri doldurarak evin sigorta masrafını tahmin edebilirsiniz.")

# 📁 Uygulama dizininden dosya yolları
model_path = "model.pkl"
columns_path = "model_columns.pkl"
file_path = "insurance_modified.csv"  # Bu dosya app ile aynı klasörde olmalı

# 📌 Kullanıcı girişleri
ev_durumu = st.selectbox("Ev Durumu", ["Ev Sahibi", "Kiralık"])
evcil_hayvan = st.selectbox("Evcil Hayvan Sahibi misiniz?", ["yes", "no"])
bolge = st.selectbox("Bölge", [
    "Batı Bölgesi", 
    "İç Anadolu ve Karadeniz Bölgesi", 
    "Akdeniz ve Ege İç Kesimleri Bölgesi", 
    "Doğu ve Güneydoğu Anadolu Bölgesi"
])
evin_yasi_kategori = st.selectbox("Evin Yaşı", [
    "0-9", "10-19", "20-29", "30-39", 
    "40-49", "50-59", "60-69", "70-79", 
    "80-89", "90-99", "100+"
])
cocuk_sayisi_kategori = st.selectbox("Çocuk Sayısı", ["Yüksek", "Normal", "Düşük"])
kozmetik_durum = st.selectbox("Evin Kozmetik Durumu", ["İyi", "Normal", "Kötü"])
teminat_bedeli = st.number_input("Evin Teminat Bedeli (₺)", min_value=1000, step=1000)

# 🎯 Girişleri DataFrame’e dönüştür
input_dict = {
    "Ev Durumu": ev_durumu,
    "Evcil Hayvan Sahibi": 1 if evcil_hayvan == "yes" else 0,
    "Bolge": bolge,
    "Katagorik_Evin_Yasi": evin_yasi_kategori,
    "Katagorik_Cocuk_Sayisi": cocuk_sayisi_kategori,
    "Katagorik_Evin_Kozmetik_Durumu": kozmetik_durum
}
input_df = pd.DataFrame([input_dict])

# 📦 Model ve kolonları yükle
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(columns_path, "rb") as f:
        model_columns = pickle.load(f)

    # 🔍 One-hot encoding
    input_processed = pd.get_dummies(input_df)

    # Eksik kolonları 0 olarak ekle
    for col in model_columns:
        if col not in input_processed.columns:
            input_processed[col] = 0
    input_processed = input_processed[model_columns]

    # 📊 Uygulama dizininden CSV dosyasını oku
    try:
        df = pd.read_csv(file_path, delimiter=";")
        toplam_masraf = df["Masraf"].sum()

        # 🎯 Hesaplama butonu
        if st.button("Beklenen Masraf Tutarını Hesapla"):
            tahmini_masraf = model.predict(input_processed)[0]
            beklenen_tutar = tahmini_masraf * teminat_bedeli

            st.write(f"**Beklenen Masraf Oranı:** {tahmini_masraf:,.4f}")
            st.success(f"💸 Beklenen Masraf Tutarı: {beklenen_tutar:,.2f} ₺")
    except FileNotFoundError:
        st.error(f"'{file_path}' dosyası bulunamadı. Lütfen bu dosyanın uygulama klasöründe olduğundan emin olun.")

except FileNotFoundError:
    st.error("Model dosyaları eksik. Lütfen 'model.pkl' ve 'model_columns.pkl' dosyalarını uygulama dizinine yükleyin.")
