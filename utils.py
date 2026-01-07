# ==============================
# UTILITY FUNCTIONS
# Yardımcı Fonksiyonlar
# ==============================

import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
from typing import List, Dict, Tuple, Union
import config

# ==============================
# LOGGING SETUP
# ==============================

def setup_logging(name: str = __name__) -> logging.Logger:
    """Logging yapılandırması"""
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format=config.LOG_FORMAT
    )
    return logging.getLogger(name)

logger = setup_logging()

# ==============================
# IMAGE PROCESSING
# ==============================

def load_and_preprocess_image(
    image_path: Union[str, Path],
    target_size: Tuple[int, int] = config.IMG_SIZE,
    preprocess_mode: str = 'mobilenet'
) -> np.ndarray:
    """
    Görüntüyü yükle ve ön işleme yap
    
    Args:
        image_path: Görüntü dosya yolu
        target_size: Hedef boyut (width, height)
        preprocess_mode: Ön işleme modu ('mobilenet', 'efficientnet', 'resnet')
    
    Returns:
        Ön işlenmiş görüntü array
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img)
        
        # Model tipine göre ön işleme
        if preprocess_mode == 'mobilenet':
            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        elif preprocess_mode == 'efficientnet':
            img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
        elif preprocess_mode == 'resnet':
            img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
        else:
            img_array = img_array / 255.0
        
        return np.expand_dims(img_array, axis=0)
    
    except Exception as e:
        logger.error(f"Görüntü yükleme hatası: {e}")
        raise

def save_image(image: np.ndarray, save_path: Union[str, Path]):
    """Görüntüyü kaydet"""
    try:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        img = Image.fromarray(image)
        img.save(save_path)
        logger.info(f"Görüntü kaydedildi: {save_path}")
    
    except Exception as e:
        logger.error(f"Görüntü kaydetme hatası: {e}")
        raise

# ==============================
# MODEL UTILITIES
# ==============================

def load_model_safe(model_path: Union[str, Path]) -> tf.keras.Model:
    """Modeli güvenli şekilde yükle"""
    try:
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Model yüklendi: {model_path}")
        return model
    except Exception as e:
        logger.error(f"Model yükleme hatası: {e}")
        raise

def get_model_info(model: tf.keras.Model) -> Dict:
    """Model bilgilerini al"""
    return {
        'total_params': model.count_params(),
        'trainable_params': sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]),
        'non_trainable_params': sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights]),
        'layers': len(model.layers),
        'input_shape': model.input_shape,
        'output_shape': model.output_shape
    }

# ==============================
# PREDICTION UTILITIES
# ==============================

def predict_single_image(
    model: tf.keras.Model,
    image_path: Union[str, Path],
    class_names: List[str] = config.CLASS_NAMES,
    top_k: int = 5
) -> Dict:
    """
    Tek bir görüntü için tahmin yap
    
    Args:
        model: Keras modeli
        image_path: Görüntü yolu
        class_names: Sınıf isimleri
        top_k: En yüksek k tahmin
    
    Returns:
        Tahmin sonuçları
    """
    img_array = load_and_preprocess_image(image_path)
    predictions = model.predict(img_array, verbose=0)[0]
    
    # En yüksek k tahmini al
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    
    results = {
        'predictions': {
            class_names[i]: float(predictions[i])
            for i in top_indices
        },
        'top_class': class_names[top_indices[0]],
        'confidence': float(predictions[top_indices[0]])
    }
    
    return results

def batch_predict(
    model: tf.keras.Model,
    image_paths: List[Union[str, Path]],
    class_names: List[str] = config.CLASS_NAMES
) -> List[Dict]:
    """Toplu tahmin"""
    results = []
    for img_path in image_paths:
        try:
            result = predict_single_image(model, img_path, class_names)
            result['image_path'] = str(img_path)
            results.append(result)
        except Exception as e:
            logger.error(f"Tahmin hatası ({img_path}): {e}")
            results.append({
                'image_path': str(img_path),
                'error': str(e)
            })
    
    return results

# ==============================
# METRICS UTILITIES
# ==============================

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Detaylı metrikler hesapla"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average='weighted')),
        'recall': float(recall_score(y_true, y_pred, average='weighted')),
        'f1_score': float(f1_score(y_true, y_pred, average='weighted'))
    }

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Union[str, Path] = None,
    figsize: Tuple[int, int] = (12, 10)
):
    """Confusion matrix çiz"""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='YlOrRd',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Tahmin Sayısı'}
    )
    plt.xlabel('Tahmin Edilen Sınıf', fontsize=12)
    plt.ylabel('Gerçek Sınıf', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        logger.info(f"Confusion matrix kaydedildi: {save_path}")
    
    plt.show()

# ==============================
# FILE UTILITIES
# ==============================

def save_json(data: Dict, file_path: Union[str, Path]):
    """JSON dosyası kaydet"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"JSON kaydedildi: {file_path}")
    except Exception as e:
        logger.error(f"JSON kaydetme hatası: {e}")
        raise

def load_json(file_path: Union[str, Path]) -> Dict:
    """JSON dosyası yükle"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"JSON yüklendi: {file_path}")
        return data
    except Exception as e:
        logger.error(f"JSON yükleme hatası: {e}")
        raise

def get_image_files(directory: Union[str, Path], extensions: List[str] = None) -> List[Path]:
    """Dizindeki görüntü dosyalarını al"""
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    directory = Path(directory)
    image_files = []
    
    for ext in extensions:
        image_files.extend(directory.glob(f'**/*{ext}'))
        image_files.extend(directory.glob(f'**/*{ext.upper()}'))
    
    return sorted(image_files)

# ==============================
# VISUALIZATION UTILITIES
# ==============================

def plot_training_history(
    history: Dict,
    save_path: Union[str, Path] = None,
    figsize: Tuple[int, int] = (16, 6)
):
    """Eğitim geçmişini görselleştir"""
    plt.figure(figsize=figsize)
    
    # Accuracy
    plt.subplot(1, 3, 1)
    plt.plot(history['accuracy'], label='Train Accuracy', linewidth=2)
    plt.plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    plt.title('Model Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Loss
    plt.subplot(1, 3, 2)
    plt.plot(history['loss'], label='Train Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Model Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Overfitting
    plt.subplot(1, 3, 3)
    train_val_diff = np.array(history['accuracy']) - np.array(history['val_accuracy'])
    plt.plot(train_val_diff, label='Train-Val Accuracy Farkı', linewidth=2, color='orange')
    plt.axhline(y=0, color='g', linestyle='--', alpha=0.5)
    plt.title('Overfitting Analizi', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy Farkı', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        logger.info(f"Eğitim grafiği kaydedildi: {save_path}")
    
    plt.show()

def display_prediction_samples(
    model: tf.keras.Model,
    image_paths: List[Path],
    class_names: List[str],
    num_samples: int = 9,
    figsize: Tuple[int, int] = (15, 15)
):
    """Örnek tahminleri görselleştir"""
    plt.figure(figsize=figsize)
    
    for i, img_path in enumerate(image_paths[:num_samples]):
        img = Image.open(img_path).convert('RGB')
        result = predict_single_image(model, img_path, class_names, top_k=1)
        
        plt.subplot(3, 3, i + 1)
        plt.imshow(img)
        plt.title(
            f"Tahmin: {result['top_class']}\nGüven: {result['confidence']*100:.1f}%",
            fontsize=10
        )
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# ==============================
# PROGRESS BAR
# ==============================

def create_progress_bar(total: int, desc: str = "Processing"):
    """Progress bar oluştur"""
    try:
        from tqdm import tqdm
        return tqdm(total=total, desc=desc)
    except ImportError:
        logger.warning("tqdm yüklü değil, progress bar gösterilmeyecek")
        return None
