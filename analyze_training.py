# ==============================
# EĞİTİM ANALİZİ VE GÖRSELLEŞTİRME
# Kapsamlı Eğitim Raporu Oluşturma
# ==============================

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from pathlib import Path
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings('ignore')

# Türkçe karakter desteği için
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("UYDU GÖRÜNTÜLERİ SINIFLANDIRMA - EĞİTİM ANALİZİ")
print("=" * 80)

# ==============================
# AYARLAR
# ==============================

DATA_DIR = r"C:\Users\Lenovo\Desktop\sinir_aglari\EuroSAT"
MODEL_PATH = r"C:\Users\Lenovo\Desktop\sinir_aglari\outputs\satellite_model.keras"
RESULTS_DIR = r"C:\Users\Lenovo\Desktop\sinir_aglari\results"

os.makedirs(RESULTS_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Sınıf isimleri ve Türkçe karşılıkları
CLASS_NAMES_TR = {
    'AnnualCrop': 'Yıllık Ekin',
    'Forest': 'Orman',
    'HerbaceousVegetation': 'Otsu Bitki Örtüsü',
    'Highway': 'Otoyol',
    'Industrial': 'Sanayi Bölgesi',
    'Pasture': 'Mera',
    'PermanentCrop': 'Kalıcı Ekin',
    'Residential': 'Yerleşim Alanı',
    'River': 'Nehir',
    'SeaLake': 'Deniz/Göl'
}

CLASS_EMOJIS = {
    'AnnualCrop': '🌾',
    'Forest': '🌲',
    'HerbaceousVegetation': '🌿',
    'Highway': '🛣️',
    'Industrial': '🏭',
    'Pasture': '🐄',
    'PermanentCrop': '🌳',
    'Residential': '🏘️',
    'River': '🌊',
    'SeaLake': '💧'
}

# ==============================
# 1. VERİ SETİ DAĞILIMI ANALİZİ
# ==============================

print("\n[1/7] Veri seti analiz ediliyor...")

class_counts = {}
for class_name in os.listdir(DATA_DIR):
    class_path = os.path.join(DATA_DIR, class_name)
    if os.path.isdir(class_path):
        count = len([f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
        class_counts[class_name] = count

# Veri seti dağılımı grafiği
plt.figure(figsize=(14, 6))
classes = list(class_counts.keys())
counts = list(class_counts.values())
colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))

bars = plt.bar(range(len(classes)), counts, color=colors, edgecolor='black', linewidth=1.5)
plt.xlabel('Sınıflar', fontsize=14, fontweight='bold')
plt.ylabel('Görüntü Sayısı', fontsize=14, fontweight='bold')
plt.title('Veri Seti Dağılımı - EuroSAT', fontsize=16, fontweight='bold', pad=20)
plt.xticks(range(len(classes)), [f"{CLASS_EMOJIS.get(c, '')} {CLASS_NAMES_TR.get(c, c)}" for c in classes], 
           rotation=45, ha='right', fontsize=10)
plt.grid(axis='y', alpha=0.3, linestyle='--')

# Her bara değer ekle
for i, (bar, count) in enumerate(zip(bars, counts)):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{count:,}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'dataset_distribution.png'), dpi=300, bbox_inches='tight')
print(f"✓ Veri seti dağılımı kaydedildi: {RESULTS_DIR}/dataset_distribution.png")
plt.close()

total_images = sum(counts)
print(f"\nToplam görüntü sayısı: {total_images:,}")
print(f"Sınıf sayısı: {len(classes)}")
print(f"Ortalama sınıf başına: {total_images//len(classes):,} görüntü")

# ==============================
# 2. MODEL VE VERİ YÜKLEME
# ==============================

print("\n[2/7] Model ve veri yükleniyor...")

# Model yükle
if not os.path.exists(MODEL_PATH):
    print(f"❌ Model bulunamadı: {MODEL_PATH}")
    print("Lütfen önce modeli eğitin: python main.py")
    exit(1)

model = load_model(MODEL_PATH)
print(f"✓ Model yüklendi: {MODEL_PATH}")

# Validation verisi yükle
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

print(f"✓ Validation verisi yüklendi: {val_data.samples} görüntü")

# ==============================
# 3. TAHMİNLER VE METRIKLER
# ==============================

print("\n[3/7] Tahminler yapılıyor...")

predictions = model.predict(val_data, verbose=1)
y_pred = np.argmax(predictions, axis=1)
y_true = val_data.classes

class_names = list(val_data.class_indices.keys())

# Sınıflandırma raporu
report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
report_text = classification_report(y_true, y_pred, target_names=class_names)

print("\n" + "="*80)
print("SINIFLANDIRMA RAPORU")
print("="*80)
print(report_text)

# ==============================
# 4. CONFUSION MATRIX
# ==============================

print("\n[4/7] Confusion matrix oluşturuluyor...")

cm = confusion_matrix(y_true, y_pred)

# Detaylı confusion matrix
plt.figure(figsize=(14, 12))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='YlOrRd',
    xticklabels=[f"{CLASS_EMOJIS.get(c, '')} {CLASS_NAMES_TR.get(c, c)}" for c in class_names],
    yticklabels=[f"{CLASS_EMOJIS.get(c, '')} {CLASS_NAMES_TR.get(c, c)}" for c in class_names],
    cbar_kws={'label': 'Tahmin Sayısı'},
    linewidths=0.5,
    linecolor='gray',
    square=True
)
plt.xlabel('Tahmin Edilen Sınıf', fontsize=14, fontweight='bold', labelpad=10)
plt.ylabel('Gerçek Sınıf', fontsize=14, fontweight='bold', labelpad=10)
plt.title('Confusion Matrix (Karmaşıklık Matrisi)', fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix_detailed.png'), dpi=300, bbox_inches='tight')
print(f"✓ Confusion matrix kaydedildi: {RESULTS_DIR}/confusion_matrix_detailed.png")
plt.close()

# ==============================
# 5. SINIF BAZINDA PERFORMANS
# ==============================

print("\n[5/7] Sınıf bazında performans grafikleri oluşturuluyor...")

# Metrikler
metrics_df = pd.DataFrame({
    'Sınıf': [CLASS_NAMES_TR.get(c, c) for c in class_names],
    'Precision': [report[c]['precision'] for c in class_names],
    'Recall': [report[c]['recall'] for c in class_names],
    'F1-Score': [report[c]['f1-score'] for c in class_names],
    'Support': [report[c]['support'] for c in class_names]
})

# Performans grafikleri
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Precision
ax1 = axes[0, 0]
bars1 = ax1.barh(metrics_df['Sınıf'], metrics_df['Precision'], color='skyblue', edgecolor='black')
ax1.set_xlabel('Precision', fontsize=12, fontweight='bold')
ax1.set_title('Sınıf Bazında Precision', fontsize=14, fontweight='bold')
ax1.set_xlim(0, 1)
ax1.grid(axis='x', alpha=0.3)
for i, v in enumerate(metrics_df['Precision']):
    ax1.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=9)

# Recall
ax2 = axes[0, 1]
bars2 = ax2.barh(metrics_df['Sınıf'], metrics_df['Recall'], color='lightcoral', edgecolor='black')
ax2.set_xlabel('Recall', fontsize=12, fontweight='bold')
ax2.set_title('Sınıf Bazında Recall', fontsize=14, fontweight='bold')
ax2.set_xlim(0, 1)
ax2.grid(axis='x', alpha=0.3)
for i, v in enumerate(metrics_df['Recall']):
    ax2.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=9)

# F1-Score
ax3 = axes[1, 0]
bars3 = ax3.barh(metrics_df['Sınıf'], metrics_df['F1-Score'], color='lightgreen', edgecolor='black')
ax3.set_xlabel('F1-Score', fontsize=12, fontweight='bold')
ax3.set_title('Sınıf Bazında F1-Score', fontsize=14, fontweight='bold')
ax3.set_xlim(0, 1)
ax3.grid(axis='x', alpha=0.3)
for i, v in enumerate(metrics_df['F1-Score']):
    ax3.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=9)

# Support
ax4 = axes[1, 1]
bars4 = ax4.barh(metrics_df['Sınıf'], metrics_df['Support'], color='plum', edgecolor='black')
ax4.set_xlabel('Örnek Sayısı', fontsize=12, fontweight='bold')
ax4.set_title('Sınıf Bazında Örnek Sayısı', fontsize=14, fontweight='bold')
ax4.grid(axis='x', alpha=0.3)
for i, v in enumerate(metrics_df['Support']):
    ax4.text(v + 10, i, f'{int(v)}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'class_performance.png'), dpi=300, bbox_inches='tight')
print(f"✓ Sınıf performansı kaydedildi: {RESULTS_DIR}/class_performance.png")
plt.close()

# ==============================
# 6. ROC EĞRİLERİ
# ==============================

print("\n[6/7] ROC eğrileri oluşturuluyor...")

# One-hot encode
y_true_bin = label_binarize(y_true, classes=range(len(class_names)))

# ROC eğrileri
plt.figure(figsize=(14, 10))
colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))

for i, (class_name, color) in enumerate(zip(class_names, colors)):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], predictions[:, i])
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color=color, lw=2, 
             label=f'{CLASS_NAMES_TR.get(class_name, class_name)} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Rastgele Tahmin')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
plt.title('ROC Eğrileri - Tüm Sınıflar', fontsize=16, fontweight='bold', pad=20)
plt.legend(loc='lower right', fontsize=9, ncol=2)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'roc_curves.png'), dpi=300, bbox_inches='tight')
print(f"✓ ROC eğrileri kaydedildi: {RESULTS_DIR}/roc_curves.png")
plt.close()

# ==============================
# 7. ÖRNEK TAHMİNLER
# ==============================

print("\n[7/7] Örnek tahminler görselleştiriliyor...")

# Doğru ve yanlış tahminleri ayır
correct_indices = np.where(y_pred == y_true)[0]
incorrect_indices = np.where(y_pred != y_true)[0]

# Rastgele örnekler seç
np.random.seed(42)
sample_correct = np.random.choice(correct_indices, min(6, len(correct_indices)), replace=False)
sample_incorrect = np.random.choice(incorrect_indices, min(6, len(incorrect_indices)), replace=False)

def plot_predictions(indices, title, filename):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, sample_idx in enumerate(indices):
        if idx >= 6:
            break
            
        # Görüntüyü al
        batch_idx = sample_idx // BATCH_SIZE
        in_batch_idx = sample_idx % BATCH_SIZE
        
        val_data.reset()
        for _ in range(batch_idx + 1):
            images, labels = next(val_data)
        
        img = images[in_batch_idx]
        true_label = class_names[y_true[sample_idx]]
        pred_label = class_names[y_pred[sample_idx]]
        confidence = predictions[sample_idx][y_pred[sample_idx]] * 100
        
        axes[idx].imshow(img)
        axes[idx].axis('off')
        
        true_tr = CLASS_NAMES_TR.get(true_label, true_label)
        pred_tr = CLASS_NAMES_TR.get(pred_label, pred_label)
        
        if y_pred[sample_idx] == y_true[sample_idx]:
            color = 'green'
            title_text = f'✓ Doğru\nGerçek: {true_tr}\nTahmin: {pred_tr}\nGüven: {confidence:.1f}%'
        else:
            color = 'red'
            title_text = f'✗ Yanlış\nGerçek: {true_tr}\nTahmin: {pred_tr}\nGüven: {confidence:.1f}%'
        
        axes[idx].set_title(title_text, fontsize=10, color=color, fontweight='bold', pad=10)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=300, bbox_inches='tight')
    print(f"✓ {title} kaydedildi: {RESULTS_DIR}/{filename}")
    plt.close()

plot_predictions(sample_correct, 'Doğru Tahmin Örnekleri', 'sample_predictions_correct.png')
plot_predictions(sample_incorrect, 'Yanlış Tahmin Örnekleri', 'sample_predictions_incorrect.png')

# ==============================
# 8. KAPSAMLI RAPOR
# ==============================

print("\n[8/8] Kapsamlı rapor oluşturuluyor...")

# Genel metrikler
accuracy = report['accuracy']
macro_avg = report['macro avg']
weighted_avg = report['weighted avg']

# Rapor metni
report_content = f"""
{'='*80}
UYDU GÖRÜNTÜLERİ SINIFLANDIRMA - KAPSAMLI EĞİTİM RAPORU
{'='*80}

Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

VERİ SETİ BİLGİLERİ
{'-'*80}
Toplam Görüntü: {total_images:,}
Sınıf Sayısı: {len(class_names)}
Validation Set: {val_data.samples:,} görüntü ({val_data.samples/total_images*100:.1f}%)

GENEL PERFORMANS METRİKLERİ
{'-'*80}
Genel Doğruluk (Accuracy): {accuracy*100:.2f}%

Macro Average:
  - Precision: {macro_avg['precision']:.4f}
  - Recall: {macro_avg['recall']:.4f}
  - F1-Score: {macro_avg['f1-score']:.4f}

Weighted Average:
  - Precision: {weighted_avg['precision']:.4f}
  - Recall: {weighted_avg['recall']:.4f}
  - F1-Score: {weighted_avg['f1-score']:.4f}

SINIF BAZINDA DETAYLI PERFORMANS
{'-'*80}
"""

for class_name in class_names:
    class_tr = CLASS_NAMES_TR.get(class_name, class_name)
    emoji = CLASS_EMOJIS.get(class_name, '')
    metrics = report[class_name]
    
    report_content += f"\n{emoji} {class_tr} ({class_name}):\n"
    report_content += f"  Precision: {metrics['precision']:.4f}\n"
    report_content += f"  Recall: {metrics['recall']:.4f}\n"
    report_content += f"  F1-Score: {metrics['f1-score']:.4f}\n"
    report_content += f"  Support: {int(metrics['support'])}\n"

report_content += f"\n{'='*80}\n"
report_content += f"DOĞRU TAHMİNLER: {len(correct_indices):,} / {len(y_true):,} ({len(correct_indices)/len(y_true)*100:.2f}%)\n"
report_content += f"YANLIŞ TAHMİNLER: {len(incorrect_indices):,} / {len(y_true):,} ({len(incorrect_indices)/len(y_true)*100:.2f}%)\n"
report_content += f"{'='*80}\n"

# Raporu kaydet
report_path = os.path.join(RESULTS_DIR, 'comprehensive_analysis_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_content)

print(f"✓ Kapsamlı rapor kaydedildi: {report_path}")

# ==============================
# ÖZET
# ==============================

print("\n" + "="*80)
print("✅ TÜM ANALİZLER TAMAMLANDI!")
print("="*80)
print(f"\n📁 Sonuçlar: {RESULTS_DIR}")
print("\nOluşturulan dosyalar:")
print("  ✓ dataset_distribution.png - Veri seti dağılımı")
print("  ✓ confusion_matrix_detailed.png - Detaylı confusion matrix")
print("  ✓ class_performance.png - Sınıf bazında performans")
print("  ✓ roc_curves.png - ROC eğrileri")
print("  ✓ sample_predictions_correct.png - Doğru tahmin örnekleri")
print("  ✓ sample_predictions_incorrect.png - Yanlış tahmin örnekleri")
print("  ✓ comprehensive_analysis_report.txt - Detaylı rapor")
print("\n" + "="*80)
print(f"\n📊 GENEL DOĞRULUK: {accuracy*100:.2f}%")
print(f"✓ Doğru: {len(correct_indices):,} / {len(y_true):,}")
print(f"✗ Yanlış: {len(incorrect_indices):,} / {len(y_true):,}")
print("="*80)
