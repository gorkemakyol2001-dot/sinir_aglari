# ==============================
# BASIT MODEL EĞİTİMİ - TEST AMAÇLI
# ==============================

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

print("=" * 60)
print("UYDU GÖRÜNTÜLERİ SINIFLANDIRMA - BASIT VERSİYON")
print("=" * 60)

# AYARLAR
DATA_DIR = r"C:\Users\Lenovo\Desktop\sinir ağları\EuroSAT"
MODEL_SAVE_PATH = "models/eurosat_simple.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5  # Daha hızlı test için

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("\n[1/6] Veri yükleniyor...")

# Basit data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    horizontal_flip=True
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
    subset='training'
)

val_data = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

print(f"\n✓ Eğitim: {train_data.samples} görüntü")
print(f"✓ Doğrulama: {val_data.samples} görüntü")
print(f"✓ Sınıflar: {list(train_data.class_indices.keys())}")

print("\n[2/6] Model oluşturuluyor...")

# Model
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(train_data.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

print("✓ Model oluşturuldu")

print("\n[3/6] Model derleniyor...")

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✓ Model derlendi")

# Callbacks
callbacks = [
    ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
]

print("\n[4/6] Eğitim başlıyor...")
print("=" * 60)

# Eğitim
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

print("\n" + "=" * 60)
print("[5/6] Eğitim tamamlandı! Grafikler oluşturuluyor...")

# Grafikler
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Val Accuracy', marker='s')
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Val Loss', marker='s')
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/training_simple.png', dpi=300)
print("✓ Grafik kaydedildi: results/training_simple.png")

print("\n[6/6] Model değerlendiriliyor...")

# Değerlendirme
val_loss, val_accuracy = model.evaluate(val_data, verbose=0)

print("\n" + "=" * 60)
print("SONUÇLAR")
print("=" * 60)
print(f"✓ Final Train Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"✓ Final Val Accuracy: {val_accuracy:.4f}")
print(f"✓ Final Train Loss: {history.history['loss'][-1]:.4f}")
print(f"✓ Final Val Loss: {val_loss:.4f}")
print(f"\n✓ Model kaydedildi: {MODEL_SAVE_PATH}")
print("=" * 60)
print("\n🎉 İŞLEM TAMAMLANDI!")
