import os
import pandas as pd
import seaborn as sns
import numpy as np
import pickle
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR, LinearSVR, NuSVR
from sklearn.neighbors import KNeighborsRegressor
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

    # **Gruplama Analizi**
    print('-' * 80)
    print("Kategorik Gruplama Analizi")
    print('-' * 80)

    kategorik_sütunlar = ['Katagorik_Evin_Kozmetik_Durumu', 'Ev Durumu', 'Evcil Hayvan Sahibi',
                          'Bolge', 'Katagorik_Evin_Yasi', 'Katagorik_Cocuk_Sayisi']

    plt.figure(figsize=(8, 5))
    sns.violinplot(x='Ev Durumu', y='Masraf', data=data)
    plt.title("Ev Durumu - Masraf")
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.violinplot(x='Evcil Hayvan Sahibi', y='Masraf', data=data)
    plt.title("Evcil Hayvan Sahibi - Masraf")
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.violinplot(x='Bolge', y='Masraf', data=data)
    plt.title("Bölge - Masraf")
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Ev Durumu', y='Masraf', data=data)
    plt.title("Ev Durumu - Masraf")
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Katagorik_Evin_Yasi', y='Masraf', data=data)
    plt.title("Evin Yaşı - Masraf")
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Katagorik_Evin_Kozmetik_Durumu', y='Masraf', data=data)
    plt.title("Evin Kozmetik Durumu - Masraf")
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Evcil Hayvan Sahibi', y='Masraf', data=data)
    plt.title("Evcil Hayvan Sahibi - Masraf")
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Katagorik_Cocuk_Sayisi', y='Masraf', data=data)
    plt.title("Çocuk Sayısı - Masraf")
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Bolge', y='Masraf', data=data)
    plt.title("Bölge - Masraf")
    plt.show()

    sns.pairplot(data, height=2)
    plt.show()

    print('Data distribution analysis')
    for v in kategorik_sütunlar:
        data = data.sort_values(by=[v])
        data[v].value_counts().plot(kind='bar')
        plt.xticks(rotation=0)
        plt.xticks(fontsize=10)
        plt.title(v)
        plt.show()

    print('Mean cost analysis:')  
    for v in kategorik_sütunlar:
        group_df = data.groupby(v)['Masraf'].mean()  # 'Masraf' sütununun ortalamasını al
        group_df.sort_values().plot(kind='bar')

        plt.xlabel(v)
        plt.ylabel('Ortalama Masraf')
        plt.xticks(rotation=0, fontsize=10)
        plt.title(f'Ortalama Sigorta Masrafı - {v}')
        plt.show()

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

# Eğitim ve test verisini ayırma
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.20, random_state=0)
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

# Model parametrelerini ve tahmin fonksiyonunu tanımlama
def train_predict_model(clf, X_train, y_train, X_test, y_test):
    ''' Fits a classifier to the training data and makes predictions '''
    print(f"Training a {clf.__class__.__name__} using a training set size of {len(X_train)}. . .")
    
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    print(f"Trained model in {end - start:.4f} seconds")
    print('#' * 50)

    # Eğitim verisi üzerinde tahmin yapma
    print("Predictions for training data:")
    start = time()
    y_pred_train = clf.predict(X_train)
    end = time()
    print(f"Made predictions for training data in {end - start:.4f} seconds.")
    print(f"R^2 score for training set: {r2_score(y_train, y_pred_train):.4f}")
    print(f"Explained variance score for training set: {explained_variance_score(y_train, y_pred_train):.4f}")
    print('#' * 50)

    # Test verisi üzerinde tahmin yapma
    print("Predictions for testing data:")
    start = time()
    y_pred_test = clf.predict(X_test)
    end = time()
    print(f"Made predictions for testing data in {end - start:.4f} seconds.")
    print(f"R^2 score for testing set: {r2_score(y_test, y_pred_test):.4f}")
    print(f"Explained variance score for testing set: {explained_variance_score(y_test, y_pred_test):.4f}")
    print('#' * 50)

# Farklı model türlerini test etme
clf_a = DecisionTreeRegressor(random_state=1)
clf_b = SVR()
clf_c = KNeighborsRegressor()
clf_d = NuSVR()

# Modelleri test etme
for clf in (clf_a, clf_b, clf_c, clf_d):
    for size in (300, 600, 900):
        train_predict_model(clf, X_train[:size], y_train[:size], X_test, y_test)
        print('-' * 80)
    print('+' * 80)

# Örnek veri ile tahmin
client_data = [
    [1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
    [0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0],
    [1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1],
    [0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0],
    [1, 0, 2, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1],
    [1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1],
]

predicted_masraf = model.predict(client_data)[0]

predictions = model.predict(client_data)
print(f"Örnek müşteri verisi ile tahmin edilen sigorta masrafları: {predictions}")