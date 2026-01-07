# ==============================
# CONFIGURATION FILE
# Merkezi Konfigürasyon Ayarları
# ==============================

import os
from pathlib import Path

# ==============================
# PATHS
# ==============================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "EuroSAT"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"
EXAMPLES_DIR = BASE_DIR / "examples"

# Dizinleri oluştur
for dir_path in [MODELS_DIR, RESULTS_DIR, LOGS_DIR, EXAMPLES_DIR]:
    dir_path.mkdir(exist_ok=True)

# ==============================
# MODEL SETTINGS
# ==============================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0001
VALIDATION_SPLIT = 0.2

# ==============================
# AVAILABLE MODELS
# ==============================

AVAILABLE_MODELS = {
    'mobilenetv2': {
        'name': 'MobileNetV2',
        'input_size': (224, 224),
        'preprocess': 'mobilenet'
    },
    'efficientnetb0': {
        'name': 'EfficientNetB0',
        'input_size': (224, 224),
        'preprocess': 'efficientnet'
    },
    'efficientnetb3': {
        'name': 'EfficientNetB3',
        'input_size': (300, 300),
        'preprocess': 'efficientnet'
    },
    'resnet50': {
        'name': 'ResNet50',
        'input_size': (224, 224),
        'preprocess': 'resnet'
    }
}

# ==============================
# CLASS NAMES
# ==============================

CLASS_NAMES = [
    'AnnualCrop',
    'Forest',
    'HerbaceousVegetation',
    'Highway',
    'Industrial',
    'Pasture',
    'PermanentCrop',
    'Residential',
    'River',
    'SeaLake'
]

CLASS_NAMES_TR = {
    'AnnualCrop': '🌾 Yıllık Ekin',
    'Forest': '🌲 Orman',
    'HerbaceousVegetation': '🌿 Otsu Bitki Örtüsü',
    'Highway': '🛣️ Otoyol',
    'Industrial': '🏭 Sanayi Bölgesi',
    'Pasture': '🐄 Mera',
    'PermanentCrop': '🌳 Kalıcı Ekin',
    'Residential': '🏘️ Yerleşim Alanı',
    'River': '🌊 Nehir',
    'SeaLake': '💧 Deniz/Göl'
}

# ==============================
# TRAINING SETTINGS
# ==============================

CALLBACKS_CONFIG = {
    'early_stopping': {
        'monitor': 'val_loss',
        'patience': 5,
        'restore_best_weights': True
    },
    'reduce_lr': {
        'monitor': 'val_loss',
        'factor': 0.5,
        'patience': 3,
        'min_lr': 1e-7
    },
    'model_checkpoint': {
        'monitor': 'val_accuracy',
        'save_best_only': True,
        'mode': 'max'
    }
}

# ==============================
# DATA AUGMENTATION
# ==============================

AUGMENTATION_CONFIG = {
    'rotation_range': 30,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'zoom_range': 0.2,
    'horizontal_flip': True,
    'vertical_flip': True,
    'fill_mode': 'nearest'
}

# ==============================
# API SETTINGS
# ==============================

API_HOST = "127.0.0.1"
API_PORT = 8000
API_TITLE = "Satellite Image Classifier API"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
🛰️ Uydu Görüntüsü Arazi Sınıflandırma API

Bu API, uydu görüntülerini 10 farklı arazi tipine sınıflandırır.

## Özellikler
- Tek görüntü tahmini
- Toplu tahmin
- Çoklu model desteği
- Ensemble tahmin
"""

# ==============================
# ENSEMBLE SETTINGS
# ==============================

ENSEMBLE_MODELS = ['mobilenetv2', 'efficientnetb0', 'resnet50']
ENSEMBLE_WEIGHTS = {
    'mobilenetv2': 0.3,
    'efficientnetb0': 0.4,
    'resnet50': 0.3
}

# ==============================
# EXPORT SETTINGS
# ==============================

EXPORT_FORMATS = ['h5', 'keras', 'tflite', 'onnx']
TFLITE_QUANTIZATION = True

# ==============================
# LOGGING
# ==============================

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
