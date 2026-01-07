# ==============================
# API CLIENT KULLANIM ÖRNEĞİ
# ==============================

import requests
import sys
sys.path.append('..')
import config
import utils
import numpy as np

print("=" * 60)
print("API CLIENT KULLANIM ÖRNEĞİ")
print("=" * 60)

API_URL = f"http://{config.API_HOST}:{config.API_PORT}"

print(f"\nAPI URL: {API_URL}")
print("\n⚠️ Not: API sunucusunun çalışıyor olması gerekir!")
print("API'yi başlatmak için: python api_server.py\n")

# Health check
print("🏥 Health check...")
try:
    response = requests.get(f"{API_URL}/health")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ API çalışıyor!")
        print(f"Yüklü modeller: {data['models_loaded']}")
        print(f"Ensemble: {'Evet' if data['ensemble_available'] else 'Hayır'}")
    else:
        print(f"❌ API yanıt vermiyor: {response.status_code}")
        exit(1)
except Exception as e:
    print(f"❌ API'ye bağlanılamadı: {e}")
    print("\nAPI sunucusunu başlatın:")
    print("  python api_server.py")
    exit(1)

# Sınıfları listele
print("\n📋 Sınıflar:")
response = requests.get(f"{API_URL}/classes")
classes = response.json()
print(f"Toplam sınıf: {classes['total_classes']}")

# Tek görüntü tahmini
print("\n" + "=" * 60)
print("TEK GÖRÜNTÜ TAHMİNİ")
print("=" * 60)

# Örnek görüntü
image_files = utils.get_image_files(config.DATA_DIR)
if not image_files:
    print("❌ Görüntü bulunamadı!")
    exit(1)

sample_image = np.random.choice(image_files)
print(f"\n🖼️ Görüntü: {sample_image}")

# Tahmin isteği
print("\n🔮 Tahmin yapılıyor...")
with open(sample_image, 'rb') as f:
    files = {'file': f}
    response = requests.post(f"{API_URL}/predict", files=files)

if response.status_code == 200:
    result = response.json()
    print(f"\n✅ Tahmin başarılı!")
    print(f"Sınıf: {result['predicted_class']}")
    print(f"Güven: {result['confidence']:.4f}")
    print(f"Model: {result['model_used']}")
    
    print(f"\n📊 Top-5 Tahminler:")
    for cls, conf in list(result['top5_predictions'].items())[:5]:
        print(f"  {cls}: {conf:.4f}")
else:
    print(f"❌ Tahmin hatası: {response.status_code}")
    print(response.text)

# Toplu tahmin
print("\n" + "=" * 60)
print("TOPLU TAHMİN")
print("=" * 60)

# 3 rastgele görüntü
sample_images = np.random.choice(image_files, size=min(3, len(image_files)), replace=False)
print(f"\n📸 {len(sample_images)} görüntü seçildi")

files = [
    ('files', (img.name, open(img, 'rb'), 'image/jpeg'))
    for img in sample_images
]

print("\n🔮 Toplu tahmin yapılıyor...")
response = requests.post(f"{API_URL}/batch_predict", files=files)

# Dosyaları kapat
for _, (_, f, _) in files:
    f.close()

if response.status_code == 200:
    result = response.json()
    print(f"\n✅ Toplu tahmin başarılı!")
    print(f"Toplam: {result['total_images']}")
    
    for pred in result['predictions']:
        if pred['success']:
            print(f"\n{pred['filename']}:")
            print(f"  Sınıf: {pred['predicted_class']}")
            print(f"  Güven: {pred['confidence']:.4f}")
        else:
            print(f"\n{pred['filename']}: ❌ Hata")
else:
    print(f"❌ Toplu tahmin hatası: {response.status_code}")

print("\n✅ API client örneği tamamlandı!")
