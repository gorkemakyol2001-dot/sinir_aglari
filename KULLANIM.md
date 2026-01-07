# 🛰️ Uydu Görüntüleri ile Arazi Sınıflandırma

## 📋 Proje Hakkında

Bu proje, uydu görüntülerinden arazi tiplerini otomatik olarak sınıflandıran bir derin öğrenme modelidir. **Transfer Learning** yöntemi ile MobileNetV2 mimarisi kullanılarak geliştirilmiştir.

### 🎯 Sınıflandırılabilen Arazi Tipleri

1. **AnnualCrop** - Yıllık Ekin (Buğday, mısır gibi)
2. **Forest** - Orman
3. **HerbaceousVegetation** - Otsu Bitki Örtüsü
4. **Highway** - Otoyol
5. **Industrial** - Sanayi Bölgesi
6. **Pasture** - Mera/Otlak
7. **PermanentCrop** - Kalıcı Ekin (Meyve bahçesi, bağ)
8. **Residential** - Yerleşim Alanı
9. **River** - Nehir
10. **SeaLake** - Deniz/Göl

### 📊 Model Performansı

- **Genel Doğruluk**: ~90%
- **Epoch Sayısı**: 10
- **Mimari**: MobileNetV2 (Transfer Learning)
- **Veri Seti**: EuroSAT

---

## 🚀 Kurulum

### Gereksinimler

```bash
pip install tensorflow numpy matplotlib seaborn pandas scikit-learn
```

### Dosya Yapısı

```
sinir ağları/
├── main.py                    # Model eğitim scripti
├── predict.py                 # Tahmin scripti
├── visualize_results.py       # Sonuç analizi
├── KULLANIM.md               # Bu dosya
├── EuroSAT/                  # Veri seti
│   ├── AnnualCrop/
│   ├── Forest/
│   └── ...
└── outputs/                   # Çıktılar (otomatik oluşur)
    ├── satellite_model.keras
    ├── training_history.csv
    ├── training_graphs.png
    ├── confusion_matrix.png
    ├── classification_report.txt
    └── analysis/
        ├── correct_predictions.png
        ├── misclassified.png
        ├── per_class_accuracy.png
        └── analysis_report.txt
```

---

## 📚 Kullanım Kılavuzu

### 1️⃣ Model Eğitimi

Modeli sıfırdan eğitmek için:

```bash
python main.py
```

**Çıktılar:**
- ✅ `outputs/satellite_model.keras` - Eğitilmiş model
- ✅ `outputs/training_history.csv` - Epoch bazında metrikler
- ✅ `outputs/training_graphs.png` - Accuracy ve Loss grafikleri
- ✅ `outputs/confusion_matrix.png` - Karmaşıklık matrisi
- ✅ `outputs/classification_report.txt` - Detaylı performans raporu

**Süre:** ~1-2 saat (GPU ile daha hızlı)

**Not:** Eğitim sırasında 3 grafik penceresi açılacaktır:
1. Training Accuracy/Loss grafikleri
2. Confusion Matrix
3. Her birini kapatarak devam edin

---

### 2️⃣ Tahmin Yapma

Eğitilmiş model ile yeni bir görüntüyü sınıflandırmak için:

#### Temel Kullanım

```bash
python predict.py --image "yol/goruntu.jpg"
```

#### Gelişmiş Kullanım

```bash
# Farklı model dosyası kullanma
python predict.py --image "yol/goruntu.jpg" --model "outputs/satellite_model.keras"

# Top-5 tahmin gösterme
python predict.py --image "yol/goruntu.jpg" --top 5
```

#### Örnek Komutlar

```bash
# Orman görüntüsü tahmini
python predict.py --image "EuroSAT/Forest/Forest_1.jpg"

# Yerleşim alanı tahmini
python predict.py --image "EuroSAT/Residential/Residential_100.jpg"

# Otoyol tahmini (top-5 sonuç)
python predict.py --image "EuroSAT/Highway/Highway_50.jpg" --top 5
```

**Çıktı Örneği:**

```
======================================================================
📊 TAHMİN SONUÇLARI (Top 3)
======================================================================

1. Forest
   📝 Açıklama: Orman
   📈 Güven: 98.45%
   🏆 EN YÜKSEK TAHMİN

2. HerbaceousVegetation
   📝 Açıklama: Otsu Bitki Örtüsü
   📈 Güven: 1.23%

3. PermanentCrop
   📝 Açıklama: Kalıcı Ekin (Meyve bahçesi, bağ)
   📈 Güven: 0.18%
======================================================================
```

Ayrıca görüntü ve tahmin grafiği otomatik olarak gösterilir.

---

### 3️⃣ Sonuçları Görselleştirme

Model performansını detaylı analiz etmek için:

```bash
python visualize_results.py
```

**Çıktılar:**

1. **Doğru Tahminler** (`correct_predictions.png`)
   - Modelin başarılı olduğu 12 rastgele örnek
   - Her görüntü için güven skoru

2. **Yanlış Tahminler** (`misclassified.png`)
   - Modelin hata yaptığı örnekler
   - Gerçek vs Tahmin edilen sınıf karşılaştırması

3. **Sınıf Bazında Performans** (`per_class_accuracy.png`)
   - Her sınıf için doğruluk oranı
   - Renk kodlu performans göstergesi:
     - 🟢 Yeşil: Mükemmel (≥90%)
     - 🟡 Turuncu: İyi (80-90%)
     - 🔴 Kırmızı: Geliştirilmeli (<80%)

4. **Detaylı Analiz Raporu** (`analysis_report.txt`)
   - Genel istatistikler
   - Sınıf bazında precision, recall, f1-score
   - En iyi/en kötü performans gösteren sınıflar
   - En çok karıştırılan sınıf çiftleri
   - İyileştirme önerileri

---

## 🔧 Parametreler ve Ayarlar

### `main.py` Ayarları

```python
DATA_DIR = r"C:\Users\Lenovo\Desktop\sinir ağları\EuroSAT"  # Veri seti yolu
OUTPUT_DIR = r"C:\Users\Lenovo\Desktop\sinir ağları\outputs"  # Çıktı klasörü

IMG_SIZE = (224, 224)  # Görüntü boyutu
BATCH_SIZE = 32        # Batch boyutu
EPOCHS = 10            # Epoch sayısı
```

**Öneriler:**
- Daha iyi performans için `EPOCHS` değerini 15-20'ye çıkarabilirsiniz
- GPU belleği yeterliyse `BATCH_SIZE` 64 yapılabilir
- Daha hızlı eğitim için `EPOCHS` azaltılabilir (ama doğruluk düşer)

---

## 📈 Model Geliştirme İpuçları

### Performansı Artırmak İçin

1. **Daha Fazla Epoch**
   ```python
   EPOCHS = 20  # main.py içinde
   ```

2. **Fine-Tuning**
   ```python
   # main.py içinde, model eğitiminden önce
   base_model.trainable = True  # Tüm katmanları eğitilebilir yap
   
   # Sadece son katmanları fine-tune et
   for layer in base_model.layers[:-20]:
       layer.trainable = False
   ```

3. **Learning Rate Ayarı**
   ```python
   # Daha düşük learning rate ile fine-tuning
   optimizer=Adam(learning_rate=0.00001)
   ```

4. **Data Augmentation Artırma**
   ```python
   datagen = ImageDataGenerator(
       rescale=1./255,
       validation_split=0.2,
       rotation_range=30,      # 20'den 30'a
       zoom_range=0.3,         # 0.2'den 0.3'e
       horizontal_flip=True,
       vertical_flip=True,     # Yeni eklendi
       brightness_range=[0.8, 1.2]  # Yeni eklendi
   )
   ```

---

## 🐛 Sorun Giderme

### Problem: Model dosyası bulunamıyor

**Hata:**
```
FileNotFoundError: outputs/satellite_model.keras
```

**Çözüm:**
```bash
# Önce modeli eğitin
python main.py
```

---

### Problem: GPU belleği yetersiz

**Hata:**
```
ResourceExhaustedError: OOM when allocating tensor
```

**Çözüm:**
```python
# main.py içinde BATCH_SIZE'ı azaltın
BATCH_SIZE = 16  # veya 8
```

---

### Problem: Veri seti bulunamıyor

**Hata:**
```
FileNotFoundError: EuroSAT directory not found
```

**Çözüm:**
```python
# main.py içinde DATA_DIR yolunu kontrol edin
DATA_DIR = r"C:\Dogru\Yol\EuroSAT"
```

---

### Problem: Tahmin yaparken hata

**Hata:**
```
ValueError: Input shape mismatch
```

**Çözüm:**
- Görüntünün geçerli bir format olduğundan emin olun (JPG, PNG)
- Görüntü dosyasının bozuk olmadığını kontrol edin

---

## 📊 Örnek Kullanım Senaryoları

### Senaryo 1: Hızlı Test

```bash
# 1. Modeli eğit (ilk kez)
python main.py

# 2. Örnek bir görüntüyü test et
python predict.py --image "EuroSAT/Forest/Forest_1.jpg"
```

---

### Senaryo 2: Detaylı Analiz

```bash
# 1. Modeli eğit
python main.py

# 2. Performans analizini çalıştır
python visualize_results.py

# 3. Analiz raporunu incele
notepad outputs/analysis/analysis_report.txt
```

---

### Senaryo 3: Toplu Tahmin

```python
# bulk_predict.py (yeni dosya oluşturun)
import os
from predict import predict_image

MODEL_PATH = "outputs/satellite_model.keras"
IMAGE_DIR = "test_images/"

for img_file in os.listdir(IMAGE_DIR):
    if img_file.endswith(('.jpg', '.png')):
        img_path = os.path.join(IMAGE_DIR, img_file)
        print(f"\n{'='*70}")
        print(f"Tahmin ediliyor: {img_file}")
        predict_image(MODEL_PATH, img_path, show_top_n=1)
```

```bash
python bulk_predict.py
```

---

## 🎓 Teknik Detaylar

### Model Mimarisi

- **Base Model**: MobileNetV2 (ImageNet ağırlıkları)
- **Eklenen Katmanlar**:
  - GlobalAveragePooling2D
  - Dense(128, activation='relu')
  - Dropout(0.3)
  - Dense(10, activation='softmax')

### Eğitim Parametreleri

- **Optimizer**: Adam (lr=0.0001)
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Data Split**: 80% Train, 20% Validation

### Data Augmentation

- Rescaling: 1/255
- Rotation: ±20°
- Zoom: ±20%
- Horizontal Flip: Evet

---

## 📞 Destek ve Katkı

### Sık Sorulan Sorular

**S: Model ne kadar sürede eğitiliyor?**
A: CPU ile ~1-2 saat, GPU ile ~20-30 dakika.

**S: Kendi veri setimi kullanabilir miyim?**
A: Evet! Veri setinizi EuroSAT ile aynı klasör yapısında organize edin.

**S: Modeli mobil uygulamada kullanabilir miyim?**
A: Evet, TensorFlow Lite'a dönüştürerek kullanabilirsiniz.

**S: Hangi görüntü formatları destekleniyor?**
A: JPG, PNG ve çoğu yaygın görüntü formatı.

---

## 📝 Lisans ve Atıf

Bu proje eğitim amaçlıdır. EuroSAT veri seti kullanılmıştır.

**EuroSAT Atıf:**
```
Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019). 
EuroSAT: A novel dataset and deep learning benchmark for land use and land cover classification. 
IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing.
```

---

## 🎉 Başarılar!

Model başarıyla eğitildi ve kullanıma hazır! Herhangi bir sorunuz varsa, lütfen iletişime geçin.

gorkemakyol2001@gmail.com

**İyi günler! 🛰️🌍**
