# ==============================
# UYDU GÖRÜNTÜLERİ İLE ARAZİ SINIFLANDIRMA
# Transfer Learning (MobileNetV2) - Geliştirilmiş Versiyon
# ==============================

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter

# ==============================
# 1. AYARLAR
# ==============================

DATA_DIR = r"C:\Users\Lenovo\Desktop\sinir ağları\EuroSAT"
MODEL_SAVE_PATH = "models/eurosat_best_model.keras"
LOG_DIR = f"logs/fit/{datetime.now().strftime('%Y%m%d-%H%M%S')}"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20  # Early stopping ile otomatik duracak
LEARNING_RATE = 0.0001

# Model dizinini oluştur
os.makedirs("models", exist_ok=True)
os.makedirs("logs/fit", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ==============================
# 2. VERİ ANALİZİ
# ==============================

print("=" * 50)
print("VERİ SETİ ANALİZİ")
print("=" * 50)

class_counts = Counter()
for class_name in os.listdir(DATA_DIR):
    class_path = os.path.join(DATA_DIR, class_name)
    if os.path.isdir(class_path):
        count = len(os.listdir(class_path))
        class_counts[class_name] = count
        print(f"{class_name}: {count} görüntü")

print(f"\nToplam Sınıf Sayısı: {len(class_counts)}")
print(f"Toplam Görüntü Sayısı: {sum(class_counts.values())}")

# Sınıf dağılımını görselleştir
plt.figure(figsize=(14, 6))
plt.bar(class_counts.keys(), class_counts.values(), color='skyblue', edgecolor='navy')
plt.xlabel('Sınıflar', fontsize=12)
plt.ylabel('Görüntü Sayısı', fontsize=12)
plt.title('Sınıf Dağılımı', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('results/class_distribution.png', dpi=300)
plt.show()

# ==============================
# 3. DATA AUGMENTATION
# ==============================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_data = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

print(f"\nEğitim Seti: {train_data.samples} görüntü")
print(f"Doğrulama Seti: {val_data.samples} görüntü")
print(f"Sınıflar: {list(train_data.class_indices.keys())}")

# ==============================
# 4. MODEL OLUŞTURMA
# ==============================

def create_model(num_classes):
    """Geliştirilmiş model mimarisi"""
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # İlk başta base model'i dondur
    base_model.trainable = False
    
    # Özel sınıflandırma katmanları
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    
    return model, base_model

model, base_model = create_model(train_data.num_classes)

# ==============================
# 5. CALLBACKS
# ==============================

callbacks = [
    # En iyi modeli kaydet
    ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    
    # Erken durdurma
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    
    # Öğrenme oranını azalt
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    
    # TensorBoard
    TensorBoard(
        log_dir=LOG_DIR,
        histogram_freq=1,
        write_graph=True,
        write_images=True
    )
]

# ==============================
# 6. MODEL DERLEME VE EĞİTİM
# ==============================

print("\n" + "=" * 50)
print("MODEL EĞİTİMİ BAŞLIYOR (1. Aşama)")
print("=" * 50)

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
)

model.summary()

# İlk eğitim (frozen base model)
history1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# ==============================
# 7. FINE-TUNING (İsteğe bağlı)
# ==============================

print("\n" + "=" * 50)
print("FINE-TUNING BAŞLIYOR (2. Aşama)")
print("=" * 50)

# Base model'in son katmanlarını aç
base_model.trainable = True

# İlk 100 katmanı dondur, sonrakileri eğit
for layer in base_model.layers[:100]:
    layer.trainable = False

# Daha düşük öğrenme oranı ile yeniden derle
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE / 10),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
)

# Fine-tuning eğitimi
history2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    callbacks=callbacks,
    verbose=1
)

# ==============================
# 8. EĞİTİM GRAFİKLERİ
# ==============================

# İki eğitim aşamasını birleştir
total_history = {
    'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
    'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
    'loss': history1.history['loss'] + history2.history['loss'],
    'val_loss': history1.history['val_loss'] + history2.history['val_loss']
}

plt.figure(figsize=(16, 6))

# Accuracy grafiği
plt.subplot(1, 3, 1)
plt.plot(total_history['accuracy'], label='Train Accuracy', linewidth=2)
plt.plot(total_history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.axvline(x=len(history1.history['accuracy']), color='r', linestyle='--', label='Fine-tuning Başlangıcı')
plt.title("Model Accuracy", fontsize=14, fontweight='bold')
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# Loss grafiği
plt.subplot(1, 3, 2)
plt.plot(total_history['loss'], label='Train Loss', linewidth=2)
plt.plot(total_history['val_loss'], label='Validation Loss', linewidth=2)
plt.axvline(x=len(history1.history['loss']), color='r', linestyle='--', label='Fine-tuning Başlangıcı')
plt.title("Model Loss", fontsize=14, fontweight='bold')
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# Overfitting analizi
plt.subplot(1, 3, 3)
train_val_diff = np.array(total_history['accuracy']) - np.array(total_history['val_accuracy'])
plt.plot(train_val_diff, label='Train-Val Accuracy Farkı', linewidth=2, color='orange')
plt.axhline(y=0, color='g', linestyle='--', alpha=0.5)
plt.axvline(x=len(history1.history['accuracy']), color='r', linestyle='--', label='Fine-tuning Başlangıcı')
plt.title("Overfitting Analizi", fontsize=14, fontweight='bold')
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Accuracy Farkı", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/training_history.png', dpi=300)
plt.show()

# ==============================
# 9. MODEL DEĞERLENDİRME
# ==============================

print("\n" + "=" * 50)
print("MODEL DEĞERLENDİRME")
print("=" * 50)

# En iyi modeli yükle
best_model = load_model(MODEL_SAVE_PATH)

# Tahminler
predictions = best_model.predict(val_data, verbose=1)
y_pred = np.argmax(predictions, axis=1)
y_true = val_data.classes

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='YlOrRd',
    xticklabels=val_data.class_indices.keys(),
    yticklabels=val_data.class_indices.keys(),
    cbar_kws={'label': 'Tahmin Sayısı'}
)
plt.xlabel("Tahmin Edilen Sınıf", fontsize=12)
plt.ylabel("Gerçek Sınıf", fontsize=12)
plt.title("Confusion Matrix", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/confusion_matrix.png', dpi=300)
plt.show()

# Classification Report
print("\n" + "=" * 50)
print("SINIFLANDIRMA RAPORU")
print("=" * 50)
report = classification_report(
    y_true,
    y_pred,
    target_names=val_data.class_indices.keys(),
    digits=4
)
print(report)

# Raporu dosyaya kaydet
with open('results/classification_report.txt', 'w', encoding='utf-8') as f:
    f.write("SINIFLANDIRMA RAPORU\n")
    f.write("=" * 50 + "\n\n")
    f.write(report)

# ==============================
# 10. ÖRNEK TAHMİNLER
# ==============================

def predict_and_visualize(model, val_data, num_samples=9):
    """Rastgele örnekler üzerinde tahmin yap ve görselleştir"""
    
    plt.figure(figsize=(15, 15))
    
    # Rastgele örnekler seç
    indices = np.random.choice(len(val_data.filenames), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        # Görüntüyü yükle
        img_path = os.path.join(DATA_DIR, val_data.filenames[idx])
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Tahmin yap
        prediction = model.predict(img_array, verbose=0)
        pred_class_idx = np.argmax(prediction)
        pred_class = list(val_data.class_indices.keys())[pred_class_idx]
        confidence = prediction[0][pred_class_idx] * 100
        
        # Gerçek sınıf
        true_class_idx = val_data.classes[idx]
        true_class = list(val_data.class_indices.keys())[true_class_idx]
        
        # Görselleştir
        plt.subplot(3, 3, i + 1)
        plt.imshow(img)
        
        color = 'green' if pred_class == true_class else 'red'
        plt.title(f"Gerçek: {true_class}\nTahmin: {pred_class}\nGüven: {confidence:.1f}%", 
                  color=color, fontsize=10)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/sample_predictions.png', dpi=300)
    plt.show()

print("\n" + "=" * 50)
print("ÖRNEK TAHMİNLER")
print("=" * 50)

predict_and_visualize(best_model, val_data, num_samples=9)

# ==============================
# 11. MODEL BİLGİLERİNİ KAYDET
# ==============================

model_info = f"""
MODEL BİLGİLERİ
{'=' * 50}

Model Adı: EuroSAT Arazi Sınıflandırma
Mimari: MobileNetV2 (Transfer Learning)
Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

VERİ SETİ:
- Toplam Görüntü: {train_data.samples + val_data.samples}
- Eğitim Seti: {train_data.samples}
- Doğrulama Seti: {val_data.samples}
- Sınıf Sayısı: {train_data.num_classes}

HİPERPARAMETRELER:
- Görüntü Boyutu: {IMG_SIZE}
- Batch Size: {BATCH_SIZE}
- Öğrenme Oranı: {LEARNING_RATE}
- Toplam Epoch: {len(total_history['accuracy'])}

SON PERFORMANS:
- Train Accuracy: {total_history['accuracy'][-1]:.4f}
- Validation Accuracy: {total_history['val_accuracy'][-1]:.4f}
- Train Loss: {total_history['loss'][-1]:.4f}
- Validation Loss: {total_history['val_loss'][-1]:.4f}

MODEL KONUMU: {MODEL_SAVE_PATH}
LOG KONUMU: {LOG_DIR}
"""

print(model_info)

with open('results/model_info.txt', 'w', encoding='utf-8') as f:
    f.write(model_info)

print("\n" + "=" * 50)
print("✅ TÜM İŞLEMLER TAMAMLANDI!")
print("=" * 50)
print(f"\n📁 Sonuçlar 'results/' klasörüne kaydedildi")
print(f"📁 Model 'models/' klasörüne kaydedildi")
print(f"📁 TensorBoard logları: {LOG_DIR}")
print(f"\nTensorBoard'u başlatmak için: tensorboard --logdir=logs/fit")
