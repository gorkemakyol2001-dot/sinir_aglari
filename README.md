# 🛰️ Uydu Görüntüleri ile Arazi Sınıflandırma

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Production--Ready-success.svg)

**Transfer Learning ile EuroSAT uydu görüntülerini sınıflandıran profesyonel derin öğrenme projesi**

[Özellikler](#-özellikler) • [Kurulum](#-kurulum) • [Kullanım](#-kullanım) • [Demo](#-demo) • [Dokümantasyon](#-dokümantasyon)

</div>

---

## 📋 İçindekiler

- [Proje Hakkında](#-proje-hakkında)
- [Özellikler](#-özellikler)
- [Kurulum](#-kurulum)
- [Hızlı Başlangıç](#-hızlı-başlangıç)
- [Kullanım Kılavuzu](#-kullanım-kılavuzu)
- [Model Performansı](#-model-performansı)
- [Dosya Yapısı](#-dosya-yapısı)
- [API Dokümantasyonu](#-api-dokümantasyonu)
- [Katkıda Bulunma](#-katkıda-bulunma)

---

## 🎯 Proje Hakkında

Bu proje, **uydu görüntülerinden arazi tiplerini otomatik olarak sınıflandıran** bir derin öğrenme sistemidir. **MobileNetV2** mimarisi ile **Transfer Learning** yöntemi kullanılarak geliştirilmiştir.

### 🌍 Sınıflandırılabilen Arazi Tipleri (10 Sınıf)

| Sınıf | Türkçe | Açıklama |
|-------|--------|----------|
| 🌾 AnnualCrop | Yıllık Ekin | Buğday, mısır gibi yıllık tarım alanları |
| 🌲 Forest | Orman | Ağaçlık orman alanları |
| 🌿 HerbaceousVegetation | Otsu Bitki Örtüsü | Çayır ve otlak alanlar |
| 🛣️ Highway | Otoyol | Karayolu ve otoyollar |
| 🏭 Industrial | Sanayi Bölgesi | Fabrika ve sanayi tesisleri |
| 🐄 Pasture | Mera | Hayvan otlatma alanları |
| 🌳 PermanentCrop | Kalıcı Ekin | Meyve bahçeleri, bağlar |
| 🏘️ Residential | Yerleşim Alanı | Konut ve yerleşim bölgeleri |
| 🌊 River | Nehir | Akarsu ve nehirler |
| 💧 SeaLake | Deniz/Göl | Su yüzeyleri |

---

## ✨ Özellikler

### 🎯 Temel Özellikler

- ✅ **Transfer Learning** - MobileNetV2 pretrained model
- ✅ **Yüksek Doğruluk** - ~90% validation accuracy
- ✅ **Data Augmentation** - Rotation, zoom, flip
- ✅ **Model Persistence** - Otomatik model kaydetme
- ✅ **Comprehensive Logging** - Detaylı eğitim logları
- ✅ **Visualization** - Training graphs, confusion matrix

### 🚀 Gelişmiş Özellikler

- 🎨 **Web Arayüzü** - Gradio ile kullanıcı dostu interface
- 🔮 **Tahmin Scripti** - Komut satırından hızlı tahmin
- 📊 **Performans Analizi** - Detaylı model değerlendirme
- 🌐 **REST API** - FastAPI ile production-ready API
- 📦 **Batch Processing** - Toplu görüntü işleme
- 💾 **Model Export** - TFLite, ONNX formatları
- 🎨 **Grad-CAM** - Model dikkat haritaları

---

## 🔧 Kurulum

### Gereksinimler

- Python 3.8+
- TensorFlow 2.x
- 8GB+ RAM (GPU önerilir)

### 1. Repository'yi Klonlayın

```bash
git clone https://github.com/yourusername/satellite-image-classification.git
cd satellite-image-classification
```

### 2. Sanal Ortam Oluşturun (Opsiyonel ama Önerilir)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

### 3. Gerekli Kütüphaneleri Yükleyin

```bash
pip install -r requirements.txt
```

### 4. Veri Setini Hazırlayın

EuroSAT veri setini [buradan](https://github.com/phelber/EuroSAT) indirin ve şu yapıda organize edin:

```
EuroSAT/
├── AnnualCrop/
├── Forest/
├── HerbaceousVegetation/
├── Highway/
├── Industrial/
├── Pasture/
├── PermanentCrop/
├── Residential/
├── River/
└── SeaLake/
```

---

## 🚀 Hızlı Başlangıç

### 1️⃣ Model Eğitimi

```bash
python main.py
```

**Çıktılar:**
- ✅ `outputs/satellite_model.keras` - Eğitilmiş model
- ✅ `outputs/training_history.csv` - Eğitim metrikleri
- ✅ `outputs/training_graphs.png` - Accuracy/Loss grafikleri
- ✅ `outputs/confusion_matrix.png` - Karmaşıklık matrisi
- ✅ `outputs/classification_report.txt` - Performans raporu

**Süre:** ~1-2 saat (CPU), ~20-30 dakika (GPU)

### 2️⃣ Web Arayüzünü Başlatın

```bash
python web_interface.py
```

Tarayıcınızda açın: `http://localhost:7860`

### 3️⃣ Tahmin Yapın

```bash
python predict.py --image "path/to/image.jpg"
```

---

## 📚 Kullanım Kılavuzu

### 🔮 Tahmin Yapma

#### Komut Satırı

```bash
# Temel kullanım
python predict.py --image "EuroSAT/Forest/Forest_1.jpg"

# Top-5 tahmin
python predict.py --image "test.jpg" --top 5

# Farklı model kullanma
python predict.py --image "test.jpg" --model "custom_model.keras"
```

**Örnek Çıktı:**

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

#### Python Kodu

```python
from predict import predict_image

# Tahmin yap
predicted_class, confidence = predict_image(
    model_path="outputs/satellite_model.keras",
    image_path="test.jpg",
    show_top_n=3
)

print(f"Sınıf: {predicted_class}")
print(f"Güven: {confidence*100:.2f}%")
```

### 📊 Performans Analizi

```bash
python visualize_results.py
```

**Oluşturulan Dosyalar:**
- `outputs/analysis/correct_predictions.png` - Başarılı tahminler
- `outputs/analysis/misclassified.png` - Hatalı tahminler
- `outputs/analysis/per_class_accuracy.png` - Sınıf bazında performans
- `outputs/analysis/analysis_report.txt` - Detaylı rapor

### 🌐 Web Arayüzü

```bash
python web_interface.py
```

**Özellikler:**
- 📸 Sürükle-bırak ile görüntü yükleme
- 🎨 Modern Gradio arayüzü
- 📊 Top-5 tahmin sonuçları
- 🖼️ Örnek görüntüler
- 🇹🇷 Türkçe dil desteği

### 🔌 REST API

#### API Sunucusunu Başlatma

```bash
python api_server.py
```

API çalışır: `http://localhost:8000`
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

#### API Kullanımı

**Python:**

```python
import requests

with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict',
        files={'file': f}
    )
    result = response.json()
    print(f"Sınıf: {result['predicted_class']}")
    print(f"Güven: {result['confidence']}")
```

**cURL:**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg"
```

### 📦 Toplu Tahmin

```bash
# CSV çıktısı
python batch_predictor.py --input_dir test_images/ --output results.csv

# JSON çıktısı
python batch_predictor.py --input_dir test_images/ --output results.json --format json
```

---

## 📊 Model Performansı

### Genel Metrikler

| Metrik | Değer |
|--------|-------|
| **Validation Accuracy** | ~90% |
| **Epoch Sayısı** | 10 |
| **Batch Size** | 32 |
| **Image Size** | 224x224 |
| **Model Boyutu** | ~14 MB |

### Sınıf Bazında Performans

| Sınıf | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| AnnualCrop | 0.90 | 0.93 | 0.91 |
| Forest | 0.90 | 0.94 | 0.92 |
| HerbaceousVegetation | 0.86 | 0.83 | 0.85 |
| Highway | 0.85 | 0.84 | 0.85 |
| Industrial | 0.93 | 0.94 | 0.93 |
| Pasture | 0.80 | 0.86 | 0.83 |
| PermanentCrop | 0.90 | 0.79 | 0.84 |
| Residential | 0.97 | 0.98 | 0.97 |
| River | 0.85 | 0.86 | 0.86 |
| SeaLake | 0.99 | 0.97 | 0.98 |

**Genel Accuracy:** 90%

---

## 📁 Dosya Yapısı

```
satellite-image-classification/
│
├── 📄 main.py                      # Ana eğitim scripti
├── 📄 predict.py                   # Tahmin scripti
├── 📄 visualize_results.py         # Performans analizi
├── 📄 web_interface.py             # Gradio web arayüzü
├── 📄 api_server.py                # FastAPI sunucusu
├── 📄 batch_predictor.py           # Toplu tahmin
├── 📄 gradcam_visualizer.py        # Grad-CAM görselleştirme
├── 📄 model_export.py              # Model dışa aktarma
├── 📄 config.py                    # Konfigürasyon
├── 📄 utils.py                     # Yardımcı fonksiyonlar
│
├── 📄 requirements.txt             # Python bağımlılıkları
├── 📄 README.md                    # Bu dosya
├── 📄 KULLANIM.md                  # Detaylı Türkçe kılavuz
│
├── 📁 EuroSAT/                     # Veri seti
│   ├── AnnualCrop/
│   ├── Forest/
│   └── ...
│
├── 📁 outputs/                     # Model ve sonuçlar
│   ├── satellite_model.keras       # Eğitilmiş model
│   ├── training_history.csv        # Eğitim metrikleri
│   ├── training_graphs.png         # Grafikler
│   ├── confusion_matrix.png        # Confusion matrix
│   ├── classification_report.txt   # Performans raporu
│   └── analysis/                   # Detaylı analiz
│       ├── correct_predictions.png
│       ├── misclassified.png
│       ├── per_class_accuracy.png
│       └── analysis_report.txt
│
├── 📁 models/                      # Ek modeller
├── 📁 logs/                        # TensorBoard logları
└── 📁 examples/                    # Kullanım örnekleri
```

---

## 🌐 API Dokümantasyonu

### Endpoints

| Method | Endpoint | Açıklama |
|--------|----------|----------|
| GET | `/` | Ana sayfa |
| GET | `/health` | Sağlık kontrolü |
| GET | `/models` | Mevcut modelleri listele |
| GET | `/classes` | Sınıfları listele |
| POST | `/predict` | Tek görüntü tahmini |
| POST | `/batch_predict` | Toplu tahmin |
| GET | `/stats` | API istatistikleri |

### Örnek Yanıt

```json
{
  "success": true,
  "predicted_class": "Forest",
  "confidence": 0.9845,
  "top5_predictions": {
    "Forest": 0.9845,
    "HerbaceousVegetation": 0.0123,
    "PermanentCrop": 0.0018,
    "AnnualCrop": 0.0009,
    "Pasture": 0.0005
  },
  "processing_time_ms": 145.3
}
```

---

## 🎓 Teknik Detaylar

### Model Mimarisi

```
Input (224x224x3)
    ↓
MobileNetV2 (Pretrained on ImageNet)
    ↓
GlobalAveragePooling2D
    ↓
Dense(128, activation='relu')
    ↓
Dropout(0.3)
    ↓
Dense(10, activation='softmax')
```

### Eğitim Parametreleri

- **Optimizer:** Adam (lr=0.0001)
- **Loss Function:** Categorical Crossentropy
- **Metrics:** Accuracy
- **Data Split:** 80% Train, 20% Validation
- **Data Augmentation:**
  - Rotation: ±20°
  - Zoom: ±20%
  - Horizontal Flip: Yes
  - Rescaling: 1/255

---

## 🛠️ Geliştirme

### Performansı Artırma

#### 1. Daha Fazla Epoch

```python
# main.py içinde
EPOCHS = 20  # 10'dan 20'ye çıkarın
```

#### 2. Fine-Tuning

```python
# Base model'in son katmanlarını eğitilebilir yapın
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False
```

#### 3. Learning Rate Ayarı

```python
optimizer=Adam(learning_rate=0.00001)  # Daha düşük LR
```

### Yeni Model Ekleme

```python
from tensorflow.keras.applications import EfficientNetB0

base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
```

---

## 📖 Dokümantasyon

- **Detaylı Kullanım Kılavuzu:** [KULLANIM.md](KULLANIM.md)
- **API Dokümantasyonu:** `http://localhost:8000/docs` (API çalışırken)
- **Kod Dokümantasyonu:** Her dosyada detaylı docstring'ler

---

## 🐛 Sorun Giderme

### Problem: Model dosyası bulunamıyor

```bash
# Önce modeli eğitin
python main.py
```

### Problem: GPU belleği yetersiz

```python
# main.py içinde batch size'ı azaltın
BATCH_SIZE = 16  # veya 8
```

### Problem: Veri seti bulunamıyor

```python
# main.py içinde DATA_DIR yolunu kontrol edin
DATA_DIR = r"C:\Dogru\Yol\EuroSAT"
```

---

## 🤝 Katkıda Bulunma

Katkılarınızı bekliyoruz! Lütfen şu adımları izleyin:

1. Fork edin
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit edin (`git commit -m 'Add amazing feature'`)
4. Push edin (`git push origin feature/amazing-feature`)
5. Pull Request açın

---

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

---

## 🙏 Teşekkürler

### Kullanılan Teknolojiler

- [TensorFlow](https://www.tensorflow.org/) - Derin öğrenme framework
- [Keras](https://keras.io/) - High-level neural networks API
- [Gradio](https://www.gradio.app/) - Web arayüzü
- [FastAPI](https://fastapi.tiangolo.com/) - REST API
- [scikit-learn](https://scikit-learn.org/) - Metrikler ve değerlendirme
- [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/) - Görselleştirme

### Veri Seti

**EuroSAT Dataset:**
```
Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019). 
EuroSAT: A novel dataset and deep learning benchmark for land use and land cover classification. 
IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing.
```

---

## 📞 İletişim

gorkemakyol2001@gmail.com

---

## 🎉 Demo

### Web Arayüzü

![Web Interface Demo](https://via.placeholder.com/800x400?text=Web+Interface+Screenshot)

### Tahmin Sonuçları

![Prediction Results](https://via.placeholder.com/800x400?text=Prediction+Results)

### Performans Grafikleri

![Performance Graphs](https://via.placeholder.com/800x400?text=Performance+Graphs)

---

<div align="center">

**⭐ Projeyi beğendiyseniz yıldız vermeyi unutmayın! ⭐**

Made with ❤️ using TensorFlow & Python

</div>
