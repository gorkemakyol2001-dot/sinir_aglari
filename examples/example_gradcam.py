# ==============================
# GRAD-CAM KULLANIM ÖRNEĞİ
# ==============================

import sys
sys.path.append('..')

from gradcam_visualizer import GradCAM
import config
import utils
import numpy as np

print("=" * 60)
print("GRAD-CAM KULLANIM ÖRNEĞİ")
print("=" * 60)

# Model yükle
model_path = config.MODELS_DIR / "eurosat_best_model.keras"

if not model_path.exists():
    print(f"❌ Model bulunamadı: {model_path}")
    print("Önce main_improved.py'yi çalıştırın!")
    exit(1)

print(f"\n📦 Model yükleniyor...")
model = utils.load_model_safe(model_path)

# Grad-CAM oluştur
gradcam = GradCAM(model)

# Örnek görüntü
image_files = utils.get_image_files(config.DATA_DIR)
sample_image = np.random.choice(image_files)

print(f"\n🖼️ Örnek görüntü: {sample_image}")

# Görselleştir
print("\n🎨 Grad-CAM görselleştirmesi oluşturuluyor...")
result = gradcam.visualize(
    sample_image,
    save_path=config.RESULTS_DIR / "example_gradcam.png"
)

print(f"\n✅ Tamamlandı!")
print(f"Tahmin: {result['predicted_class']}")
print(f"Güven: {result['confidence']:.2f}%")
print(f"Kaydedildi: {config.RESULTS_DIR / 'example_gradcam.png'}")
