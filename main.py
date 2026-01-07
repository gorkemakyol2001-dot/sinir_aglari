# ==============================
# UYDU GÖRÜNTÜLERİ İLE ARAZİ SINIFLANDIRMA
# Transfer Learning (MobileNetV2)
# ==============================

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from datetime import datetime

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import confusion_matrix, classification_report

# ==============================
# 1. AYARLAR
# ==============================

DATA_DIR = r"C:\Users\Lenovo\Desktop\sinir ağları\EuroSAT"
OUTPUT_DIR = r"C:\Users\Lenovo\Desktop\sinir ağları\outputs"

# Outputs klasörünü oluştur
os.makedirs(OUTPUT_DIR, exist_ok=True)
   
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

print(f"🚀 Eğitim başlıyor...")
print(f"📁 Sonuçlar: {OUTPUT_DIR}")

# ==============================
# 2. DATA AUGMENTATION
# ==============================

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Shuffle olmadan validation verisi (değerlendirme için)
val_data_no_shuffle = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# ==============================
# 3. MODEL (TRANSFER LEARNING)
# ==============================

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False  # pretrained ağı donduruyoruz

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(train_data.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# ==============================
# 4. MODEL DERLEME
# ==============================

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==============================
# 5. MODEL EĞİTİMİ
# ==============================

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# ==============================
# 6. MODEL KAYDETME
# ==============================

model_path = os.path.join(OUTPUT_DIR, "satellite_model.keras")
model.save(model_path)
print(f"\n✅ Model kaydedildi: {model_path}")

# ==============================
# 7. EĞİTİM GEÇMİŞİNİ KAYDET
# ==============================

history_df = pd.DataFrame(history.history)
history_df['epoch'] = range(1, len(history_df) + 1)
history_csv = os.path.join(OUTPUT_DIR, "training_history.csv")
history_df.to_csv(history_csv, index=False)
print(f"✅ Eğitim geçmişi kaydedildi: {history_csv}")

# ==============================
# 8. EĞİTİM GRAFİKLERİ
# ==============================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
plt.title("Model Accuracy", fontsize=14, fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='s')
plt.title("Model Loss", fontsize=14, fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
graph_path = os.path.join(OUTPUT_DIR, "training_graphs.png")
plt.savefig(graph_path, dpi=300, bbox_inches='tight')
print(f"✅ Eğitim grafikleri kaydedildi: {graph_path}")
plt.show()

# ==============================
# 9. CONFUSION MATRIX
# ==============================

print("\n📊 Tahminler yapılıyor...")
predictions = model.predict(val_data_no_shuffle)
y_pred = np.argmax(predictions, axis=1)

cm = confusion_matrix(val_data_no_shuffle.classes, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=val_data_no_shuffle.class_indices.keys(),
    yticklabels=val_data_no_shuffle.class_indices.keys(),
    cbar_kws={'label': 'Sayı'}
)
plt.xlabel("Tahmin Edilen Sınıf", fontsize=12)
plt.ylabel("Gerçek Sınıf", fontsize=12)
plt.title("Confusion Matrix (Karmaşıklık Matrisi)", fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
print(f"✅ Confusion matrix kaydedildi: {cm_path}")
plt.show()

# ==============================
# 10. SINIFLANDIRMA RAPORU
# ==============================

report = classification_report(
    val_data_no_shuffle.classes,
    y_pred,
    target_names=val_data_no_shuffle.class_indices.keys()
)

print("\n" + "="*60)
print("SINIFLANDIRMA RAPORU")
print("="*60)
print(report)

# Raporu dosyaya kaydet
report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("UYDU GÖRÜNTÜLERİ SINIFLANDIRMA RAPORU\n")
    f.write(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("="*60 + "\n\n")
    f.write(report)
    f.write("\n\nModel Bilgileri:\n")
    f.write(f"- Epoch Sayısı: {EPOCHS}\n")
    f.write(f"- Batch Size: {BATCH_SIZE}\n")
    f.write(f"- Image Size: {IMG_SIZE}\n")
    f.write(f"- Model: MobileNetV2 (Transfer Learning)\n")

print(f"✅ Sınıflandırma raporu kaydedildi: {report_path}")

print("\n" + "="*60)
print("🎉 EĞİTİM TAMAMLANDI!")
print("="*60)
print(f"\n📁 Tüm sonuçlar şurada: {OUTPUT_DIR}")
print("\nKaydedilen dosyalar:")
print(f"  ✓ Model: satellite_model.keras")
print(f"  ✓ Eğitim geçmişi: training_history.csv")
print(f"  ✓ Grafikler: training_graphs.png")
print(f"  ✓ Confusion Matrix: confusion_matrix.png")
print(f"  ✓ Rapor: classification_report.txt")
print("\n💡 Tahmin yapmak için: python predict.py --image <resim_yolu>")
print("="*60)
