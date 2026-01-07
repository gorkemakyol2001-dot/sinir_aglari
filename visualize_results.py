# ==============================
# SONUÇLARI GÖRSELLEŞTİRME VE ANALİZ
# Model performansını detaylı analiz eder
# ==============================

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

# ==============================
# AYARLAR
# ==============================

DATA_DIR = r"C:\Users\Lenovo\Desktop\sinir ağları\EuroSAT"
MODEL_PATH = r"C:\Users\Lenovo\Desktop\sinir ağları\outputs\satellite_model.keras"
OUTPUT_DIR = r"C:\Users\Lenovo\Desktop\sinir ağları\outputs\analysis"

# Analysis klasörünü oluştur
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

print("="*70)
print("📊 MODEL PERFORMANS ANALİZİ")
print("="*70)

# ==============================
# VERİ YÜKLEME
# ==============================

print("\n🔄 Veri yükleniyor...")

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

val_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

print(f"✅ {val_data.samples} validation görüntüsü yüklendi")
print(f"📁 Sınıf sayısı: {val_data.num_classes}")

# ==============================
# MODEL YÜKLEME
# ==============================

print(f"\n🔄 Model yükleniyor: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model yüklendi!")

# ==============================
# TAHMİNLER
# ==============================

print("\n🔮 Tahminler yapılıyor...")
predictions = model.predict(val_data, verbose=1)
y_pred = np.argmax(predictions, axis=1)
y_true = val_data.classes

print("✅ Tahminler tamamlandı!")

# ==============================
# SINIF İSİMLERİ
# ==============================

class_names = list(val_data.class_indices.keys())

# ==============================
# 1. DOĞRU TAHMİNLER
# ==============================

print("\n📸 Doğru tahminleri görselleştiriliyor...")

correct_indices = np.where(y_pred == y_true)[0]
np.random.shuffle(correct_indices)

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
fig.suptitle('✅ DOĞRU TAHMİNLER (Rastgele Örnekler)', fontsize=16, fontweight='bold')

for i, ax in enumerate(axes.flat):
    if i < len(correct_indices):
        idx = correct_indices[i]
        
        # Görüntüyü al
        img_path = val_data.filepaths[idx]
        img = plt.imread(img_path)
        
        true_label = class_names[y_true[idx]]
        confidence = predictions[idx][y_pred[idx]] * 100
        
        ax.imshow(img)
        ax.set_title(f'{true_label}\nGüven: {confidence:.1f}%', 
                    fontsize=10, color='green', fontweight='bold')
        ax.axis('off')
    else:
        ax.axis('off')

plt.tight_layout()
correct_path = os.path.join(OUTPUT_DIR, "correct_predictions.png")
plt.savefig(correct_path, dpi=300, bbox_inches='tight')
print(f"✅ Kaydedildi: {correct_path}")
plt.show()

# ==============================
# 2. YANLIŞ TAHMİNLER
# ==============================

print("\n❌ Yanlış tahminleri görselleştiriliyor...")

wrong_indices = np.where(y_pred != y_true)[0]
np.random.shuffle(wrong_indices)

if len(wrong_indices) > 0:
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('❌ YANLIŞ TAHMİNLER (Hata Analizi)', fontsize=16, fontweight='bold', color='red')
    
    for i, ax in enumerate(axes.flat):
        if i < len(wrong_indices) and i < 12:
            idx = wrong_indices[i]
            
            # Görüntüyü al
            img_path = val_data.filepaths[idx]
            img = plt.imread(img_path)
            
            true_label = class_names[y_true[idx]]
            pred_label = class_names[y_pred[idx]]
            confidence = predictions[idx][y_pred[idx]] * 100
            
            ax.imshow(img)
            ax.set_title(f'Gerçek: {true_label}\nTahmin: {pred_label}\nGüven: {confidence:.1f}%', 
                        fontsize=9, color='red', fontweight='bold')
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    wrong_path = os.path.join(OUTPUT_DIR, "misclassified.png")
    plt.savefig(wrong_path, dpi=300, bbox_inches='tight')
    print(f"✅ Kaydedildi: {wrong_path}")
    plt.show()
else:
    print("🎉 Hiç yanlış tahmin yok! (Mükemmel performans)")

# ==============================
# 3. SINIF BAZINDA DOĞRULUK
# ==============================

print("\n📊 Sınıf bazında performans hesaplanıyor...")

class_accuracy = {}
for i, class_name in enumerate(class_names):
    class_indices = np.where(y_true == i)[0]
    if len(class_indices) > 0:
        correct = np.sum(y_pred[class_indices] == y_true[class_indices])
        accuracy = (correct / len(class_indices)) * 100
        class_accuracy[class_name] = accuracy

# Sıralı bar grafiği
sorted_classes = sorted(class_accuracy.items(), key=lambda x: x[1], reverse=True)
classes = [x[0] for x in sorted_classes]
accuracies = [x[1] for x in sorted_classes]

plt.figure(figsize=(12, 8))
colors = ['#2ecc71' if acc >= 90 else '#f39c12' if acc >= 80 else '#e74c3c' for acc in accuracies]
bars = plt.barh(classes, accuracies, color=colors)

plt.xlabel('Doğruluk (%)', fontsize=12, fontweight='bold')
plt.title('Sınıf Bazında Model Performansı', fontsize=14, fontweight='bold')
plt.xlim(0, 100)

# Bar üzerine değerleri yaz
for bar, acc in zip(bars, accuracies):
    width = bar.get_width()
    plt.text(width + 1, bar.get_y() + bar.get_height()/2, 
            f'{acc:.1f}%', 
            ha='left', va='center', fontweight='bold', fontsize=10)

# Renk açıklaması
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ecc71', label='Mükemmel (≥90%)'),
    Patch(facecolor='#f39c12', label='İyi (80-90%)'),
    Patch(facecolor='#e74c3c', label='Geliştirilmeli (<80%)')
]
plt.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
class_acc_path = os.path.join(OUTPUT_DIR, "per_class_accuracy.png")
plt.savefig(class_acc_path, dpi=300, bbox_inches='tight')
print(f"✅ Kaydedildi: {class_acc_path}")
plt.show()

# ==============================
# 4. DETAYLI ANALİZ RAPORU
# ==============================

print("\n📝 Detaylı analiz raporu oluşturuluyor...")

report = classification_report(y_true, y_pred, target_names=class_names, digits=4)

analysis_text = f"""
{'='*70}
UYDU GÖRÜNTÜLERİ SINIFLANDIRMA - DETAYLI PERFORMANS ANALİZİ
{'='*70}

📊 GENEL İSTATİSTİKLER
{'='*70}
Toplam Test Görüntüsü: {len(y_true)}
Doğru Tahmin: {np.sum(y_pred == y_true)}
Yanlış Tahmin: {np.sum(y_pred != y_true)}
Genel Doğruluk: {(np.sum(y_pred == y_true) / len(y_true) * 100):.2f}%

{'='*70}
SINIF BAZINDA PERFORMANS
{'='*70}

{report}

{'='*70}
SINIF BAZINDA DOĞRULUK ORANLARI
{'='*70}
"""

for class_name, acc in sorted_classes:
    status = "🟢 Mükemmel" if acc >= 90 else "🟡 İyi" if acc >= 80 else "🔴 Geliştirilmeli"
    analysis_text += f"\n{class_name:25s} {acc:6.2f}%  {status}"

analysis_text += f"""

{'='*70}
EN İYİ PERFORMANS GÖSTEREN SINIFLAR
{'='*70}
"""

for i, (class_name, acc) in enumerate(sorted_classes[:3], 1):
    analysis_text += f"\n{i}. {class_name}: {acc:.2f}%"

analysis_text += f"""

{'='*70}
GELİŞTİRİLMESİ GEREKEN SINIFLAR
{'='*70}
"""

for i, (class_name, acc) in enumerate(sorted_classes[-3:], 1):
    analysis_text += f"\n{i}. {class_name}: {acc:.2f}%"

analysis_text += f"""

{'='*70}
CONFUSION MATRIX ANALİZİ
{'='*70}

En çok karıştırılan sınıf çiftleri:
"""

cm = confusion_matrix(y_true, y_pred)
confusion_pairs = []

for i in range(len(class_names)):
    for j in range(len(class_names)):
        if i != j and cm[i][j] > 0:
            confusion_pairs.append((class_names[i], class_names[j], cm[i][j]))

confusion_pairs.sort(key=lambda x: x[2], reverse=True)

for i, (true_class, pred_class, count) in enumerate(confusion_pairs[:5], 1):
    analysis_text += f"\n{i}. {true_class} → {pred_class}: {count} kez"

analysis_text += f"""

{'='*70}
ÖNERİLER
{'='*70}

"""

# Öneriler
if len(wrong_indices) == 0:
    analysis_text += "✅ Model mükemmel performans gösteriyor!\n"
elif (np.sum(y_pred == y_true) / len(y_true)) >= 0.90:
    analysis_text += "✅ Model çok iyi performans gösteriyor.\n"
    analysis_text += "💡 Düşük performanslı sınıflar için daha fazla veri toplanabilir.\n"
else:
    analysis_text += "⚠️  Model performansı geliştirilebilir.\n"
    analysis_text += "💡 Öneriler:\n"
    analysis_text += "   - Daha fazla epoch ile eğitim\n"
    analysis_text += "   - Data augmentation artırılabilir\n"
    analysis_text += "   - Fine-tuning uygulanabilir\n"

analysis_text += f"\n{'='*70}\n"

# Dosyaya kaydet
report_path = os.path.join(OUTPUT_DIR, "analysis_report.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(analysis_text)

print(f"✅ Kaydedildi: {report_path}")

# Konsola yazdır
print(analysis_text)

print("\n" + "="*70)
print("🎉 ANALİZ TAMAMLANDI!")
print("="*70)
print(f"\n📁 Tüm analiz sonuçları: {OUTPUT_DIR}")
print("\nOluşturulan dosyalar:")
print("  ✓ correct_predictions.png - Doğru tahminler")
print("  ✓ misclassified.png - Yanlış tahminler")
print("  ✓ per_class_accuracy.png - Sınıf bazında performans")
print("  ✓ analysis_report.txt - Detaylı analiz raporu")
print("="*70 + "\n")
