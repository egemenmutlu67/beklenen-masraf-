import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, explained_variance_score
from time import time

# Kullanıcının masaüstü dizinini alma
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

# Dosya yolu
file_path = os.path.join(desktop_path, "insurance_modified.csv")

# CSV dosyasını oku
if os.path.exists(file_path):
    data = pd.read_csv(file_path, delimiter=";")  # Noktalı virgülle ayrılmış olabilir

    # Eksik verileri temizleme
    data = data.dropna()

    # Veri seti hakkında bilgi
    print(data.info())
    print(data.head(5))
    print('-' * 90)
    print(f"Veri başarıyla yüklendi. Veri kümesi {data.shape[0]} satır ve {data.shape[1]} sütun içeriyor.")

    # **Evin Kozmetik Durumu Kategorik Hale Getirilmesi**
    def ekd_category(kozmetik_durum):
        if kozmetik_durum < 19.9:
            return 'İyi'
        elif 19.9 <= kozmetik_durum <= 30:
            return 'Normal'
        else:
            return 'Kötü'

    # **Evin Yaşı Kategorik Hale Getirilmesi**
    def age_category(evin_yasi):
        age_dict = {
            0: '0-9',
            1: '10-19',
            2: '20-29',
            3: '30-39',
            4: '40-49',
            5: '50-59',
            6: '60-69',
            7: '70-79',
            8: '80-89',
            9: '90-99',
            10: '100+'
        }
        return age_dict[evin_yasi // 10]

    # **Çocuk Sayısı Kategorik Hale Getirilmesi**
    def cocuk_category(cocuk_sayisi):
        if cocuk_sayisi < 2:
            return 'Yüksek'
        elif cocuk_sayisi == 2:
            return 'Normal'
        else:
            return 'Düşük'

    # Yeni sütunları oluştur
    data['Katagorik_Evin_Kozmetik_Durumu'] = data['Evin Kozmetik Durumu'].apply(ekd_category)
    data['Katagorik_Evin_Yasi'] = data['Evin Yasi'].apply(age_category)
    data['Katagorik_Cocuk_Sayisi'] = data['Cocuk Sayisi'].apply(cocuk_category)

    # Hedef ve özelliklerin ayrılması
    target = data['Masraf']
    features = data.drop(['Evin Yasi', 'Evin Kozmetik Durumu', 'Cocuk Sayisi', 'Masraf'], axis=1)

    # Kategorik verileri sayısal verilere dönüştürme
    output = pd.DataFrame(index=features.index)
    for col, col_data in features.items():
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix=col)
        output = output.join(col_data)

    features = output
    print(f"Processed feature columns ({len(features.columns)} total features):\n{list(features.columns)}")

    # Özellikleri normalleştirme (0-1 arası)
    scaler = MinMaxScaler()
    features_norm = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

    # **Masraf'ı normalleştiriyoruz**
    target_scaler = MinMaxScaler()
    target_scaled = target.values.reshape(-1, 1)  # Hedefi bir sütun vektörüne çeviriyoruz
    target_scaled = target_scaler.fit_transform(target_scaled).flatten()  # Normalleştiriyoruz

    # Eğitim ve test verisini ayırma
    X_train, X_test, y_train, y_test = train_test_split(features_norm, target_scaled, test_size=0.20, random_state=0)
    print("Training and testing split was successful.")
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")

    # Modeli eğitme
    model = DecisionTreeRegressor(random_state=0)
    model.fit(X_train, y_train)

    # Modeli kaydetme
    with open("model.pkl", "wb") as model_file:
        pickle.dump(model, model_file)

    # Kolonları kaydetme
    with open("model_columns.pkl", "wb") as cols_file:
        pickle.dump(features.columns.tolist(), cols_file)

    # **Yeni veri ile tahmin yapmak**
    # Yeni veri (client_data) için doğru özellikleri oluşturma
    client_data = [
        [1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
        [0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0],
        [1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1],
        [0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0],
        [1, 0, 2, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1],
        [1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1],
    ]

    # client_data'ya doğru sütunları ekleyip uyumlu hale getirme
    client_data_df = pd.DataFrame(client_data, columns=features_norm.columns)

    # Eğitim verisinin kategorik sütunlarını aynı şekilde dönüştürme
    client_data_df = pd.get_dummies(client_data_df, drop_first=True)

    # Eğitim verisiyle aynı sütunlara sahip olmak için eksik sütunları eklemek
    missing_cols = set(features_norm.columns) - set(client_data_df.columns)
    for col in missing_cols:
        client_data_df[col] = 0

    # Sıra, sütun sırasının aynı olduğundan emin olmak
    client_data_df = client_data_df[features_norm.columns]

    # Normalleştirilmiş veri ile tahmin yapma
    predicted_masraf_scaled = model.predict(client_data_df)

    # **Normalleştirilmiş tahminleri orijinal aralığa geri dönüştürme**
    predicted_masraf_original = target_scaler.inverse_transform(predicted_masraf_scaled.reshape(-1, 1))

    # Tahminler
    print(f"Normalleştirilmiş tahminler: {predicted_masraf_scaled}")
    print(f"Orijinal tahminler: {predicted_masraf_original}")
