# ==============================
# ENSEMBLE KULLANIM ÖRNEĞİ
# ==============================

import sys
sys.path.append('..')

from ensemble_predictor import EnsemblePredictor
import config
import utils
import numpy as np

print("=" * 60)
print("ENSEMBLE KULLANIM ÖRNEĞİ")
print("=" * 60)

# Ensemble oluştur
print("\n📦 Ensemble predictor oluşturuluyor...")
try:
    ensemble = EnsemblePredictor(
        weights=config.ENSEMBLE_WEIGHTS,
        strategy='weighted_average'
    )
except Exception as e:
    print(f"❌ Hata: {e}")
    print("\nÖnce modelleri eğitin:")
    print("  python multi_model_trainer.py")
    exit(1)

# Örnek görüntü
image_files = utils.get_image_files(config.DATA_DIR)
sample_image = np.random.choice(image_files)

print(f"\n🖼️ Örnek görüntü: {sample_image}")

# Tahmin
print("\n🔮 Ensemble tahmin yapılıyor...")
result = ensemble.predict(sample_image, return_all=True)

print(f"\n✅ Tamamlandı!")
print(f"\n🎯 Ensemble Tahmini:")
print(f"  Sınıf: {result['ensemble_prediction']['class']}")
print(f"  Güven: {result['ensemble_prediction']['confidence']:.4f}")

print(f"\n📊 Top-5 Tahminler:")
for cls, conf in list(result['ensemble_prediction']['top5'].items())[:5]:
    print(f"  {cls}: {conf:.4f}")

if 'individual_predictions' in result:
    print(f"\n🤖 Bireysel Model Tahminleri:")
    for model_name, pred in result['individual_predictions'].items():
        print(f"  {model_name}: {pred['class']} ({pred['confidence']:.4f})")

# Strateji karşılaştırması
print("\n" + "=" * 60)
print("STRATEJİ KARŞILAŞTIRMASI")
print("=" * 60)

strategy_results = ensemble.compare_strategies(sample_image)

for strategy, pred in strategy_results.items():
    if 'error' not in pred:
        print(f"\n{strategy}:")
        print(f"  Sınıf: {pred['class']}")
        print(f"  Güven: {pred['confidence']:.4f}")
