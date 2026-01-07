# ==============================
# GRAD-CAM VISUALIZER
# Gradient-weighted Class Activation Mapping
# ==============================

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from pathlib import Path
from typing import Union, Tuple
import config
import utils

logger = utils.setup_logging(__name__)

# ==============================
# GRAD-CAM IMPLEMENTATION
# ==============================

class GradCAM:
    """Grad-CAM görselleştirme sınıfı"""
    
    def __init__(self, model: tf.keras.Model, layer_name: str = None):
        """
        Args:
            model: Keras modeli
            layer_name: Aktivasyon haritası için katman adı (None ise son conv katmanı)
        """
        self.model = model
        self.layer_name = layer_name or self._find_last_conv_layer()
        
        logger.info(f"Grad-CAM katmanı: {self.layer_name}")
    
    def _find_last_conv_layer(self) -> str:
        """Son convolutional katmanı bul"""
        for layer in reversed(self.model.layers):
            if 'conv' in layer.name.lower():
                return layer.name
        
        raise ValueError("Model'de convolutional katman bulunamadı!")
    
    def generate_heatmap(
        self,
        image: np.ndarray,
        class_idx: int = None,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Grad-CAM ısı haritası oluştur
        
        Args:
            image: Girdi görüntüsü (preprocessed)
            class_idx: Hedef sınıf indeksi (None ise en yüksek tahmin)
            normalize: Haritayı normalize et
        
        Returns:
            Isı haritası (0-1 arası)
        """
        # Gradient model oluştur
        grad_model = tf.keras.models.Model(
            inputs=self.model.input,
            outputs=[
                self.model.get_layer(self.layer_name).output,
                self.model.output
            ]
        )
        
        # Gradient hesapla
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image)
            
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            
            class_channel = predictions[:, class_idx]
        
        # Gradientleri al
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Global average pooling
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weighted combination
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize
        if normalize:
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()
    
    def overlay_heatmap(
        self,
        heatmap: np.ndarray,
        original_image: np.ndarray,
        alpha: float = 0.4,
        colormap: str = 'jet'
    ) -> np.ndarray:
        """
        Isı haritasını orijinal görüntü üzerine bindir
        
        Args:
            heatmap: Grad-CAM ısı haritası
            original_image: Orijinal görüntü
            alpha: Şeffaflık (0-1)
            colormap: Renk haritası
        
        Returns:
            Overlay görüntü
        """
        # Heatmap'i resize et
        heatmap = np.uint8(255 * heatmap)
        
        # Colormap uygula
        cmap = cm.get_cmap(colormap)
        heatmap_colored = cmap(heatmap)[:, :, :3]
        heatmap_colored = np.uint8(255 * heatmap_colored)
        
        # Resize to original image size
        heatmap_resized = Image.fromarray(heatmap_colored).resize(
            (original_image.shape[1], original_image.shape[0]),
            Image.BILINEAR
        )
        heatmap_resized = np.array(heatmap_resized)
        
        # Overlay
        overlay = heatmap_resized * alpha + original_image * (1 - alpha)
        overlay = np.uint8(overlay)
        
        return overlay
    
    def visualize(
        self,
        image_path: Union[str, Path],
        class_idx: int = None,
        save_path: Union[str, Path] = None,
        figsize: Tuple[int, int] = (15, 5)
    ):
        """
        Grad-CAM görselleştirmesi oluştur ve göster
        
        Args:
            image_path: Görüntü dosya yolu
            class_idx: Hedef sınıf (None ise en yüksek tahmin)
            save_path: Kayıt yolu
            figsize: Figure boyutu
        """
        # Orijinal görüntüyü yükle
        original_img = Image.open(image_path).convert('RGB')
        original_img = original_img.resize(config.IMG_SIZE)
        original_array = np.array(original_img)
        
        # Preprocessed görüntü
        preprocessed = utils.load_and_preprocess_image(image_path)
        
        # Tahmin yap
        predictions = self.model.predict(preprocessed, verbose=0)[0]
        
        if class_idx is None:
            class_idx = np.argmax(predictions)
        
        predicted_class = config.CLASS_NAMES[class_idx]
        confidence = predictions[class_idx] * 100
        
        # Heatmap oluştur
        heatmap = self.generate_heatmap(preprocessed, class_idx)
        
        # Heatmap'i resize et
        heatmap_resized = np.array(
            Image.fromarray(np.uint8(255 * heatmap)).resize(
                config.IMG_SIZE,
                Image.BILINEAR
            )
        ) / 255.0
        
        # Overlay oluştur
        overlay = self.overlay_heatmap(heatmap_resized, original_array)
        
        # Görselleştir
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Orijinal görüntü
        axes[0].imshow(original_array)
        axes[0].set_title('Orijinal Görüntü', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Heatmap
        axes[1].imshow(heatmap_resized, cmap='jet')
        axes[1].set_title('Grad-CAM Haritası', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(overlay)
        axes[2].set_title(
            f'Overlay\n{predicted_class} ({confidence:.1f}%)',
            fontsize=12,
            fontweight='bold'
        )
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Grad-CAM görselleştirmesi kaydedildi: {save_path}")
        
        plt.show()
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'heatmap': heatmap_resized,
            'overlay': overlay
        }
    
    def batch_visualize(
        self,
        image_paths: list,
        save_dir: Union[str, Path] = None,
        num_samples: int = 9
    ):
        """Birden fazla görüntü için Grad-CAM"""
        save_dir = Path(save_dir) if save_dir else config.RESULTS_DIR / "gradcam"
        save_dir.mkdir(exist_ok=True, parents=True)
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.flatten()
        
        for i, img_path in enumerate(image_paths[:num_samples]):
            # Görüntüyü yükle
            original_img = Image.open(img_path).convert('RGB')
            original_img = original_img.resize(config.IMG_SIZE)
            original_array = np.array(original_img)
            
            # Preprocessed
            preprocessed = utils.load_and_preprocess_image(img_path)
            
            # Tahmin
            predictions = self.model.predict(preprocessed, verbose=0)[0]
            class_idx = np.argmax(predictions)
            predicted_class = config.CLASS_NAMES[class_idx]
            confidence = predictions[class_idx] * 100
            
            # Heatmap
            heatmap = self.generate_heatmap(preprocessed, class_idx)
            heatmap_resized = np.array(
                Image.fromarray(np.uint8(255 * heatmap)).resize(
                    config.IMG_SIZE,
                    Image.BILINEAR
                )
            ) / 255.0
            
            # Overlay
            overlay = self.overlay_heatmap(heatmap_resized, original_array)
            
            # Görselleştir
            axes[i].imshow(overlay)
            axes[i].set_title(
                f'{predicted_class}\n{confidence:.1f}%',
                fontsize=10
            )
            axes[i].axis('off')
        
        plt.tight_layout()
        
        save_path = save_dir / "batch_gradcam.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Batch Grad-CAM kaydedildi: {save_path}")
        
        plt.show()

# ==============================
# MAIN EXECUTION
# ==============================

if __name__ == "__main__":
    print("=" * 60)
    print("GRAD-CAM GÖRSELLEŞTİRME")
    print("=" * 60)
    
    # Model yükle
    model_path = config.MODELS_DIR / "eurosat_best_model.keras"
    
    if not model_path.exists():
        print(f"❌ Model bulunamadı: {model_path}")
        print("Önce main_improved.py'yi çalıştırarak modeli eğitin!")
        exit(1)
    
    print(f"\n📦 Model yükleniyor: {model_path}")
    model = utils.load_model_safe(model_path)
    
    # Grad-CAM oluştur
    gradcam = GradCAM(model)
    
    # Örnek görüntüler
    print("\n🔍 Örnek görüntüler aranıyor...")
    image_files = utils.get_image_files(config.DATA_DIR)
    
    if not image_files:
        print("❌ Görüntü bulunamadı!")
        exit(1)
    
    print(f"✅ {len(image_files)} görüntü bulundu")
    
    # Tek görüntü için Grad-CAM
    print("\n" + "=" * 60)
    print("TEK GÖRÜNTÜ GRAD-CAM")
    print("=" * 60)
    
    sample_image = np.random.choice(image_files)
    print(f"Görüntü: {sample_image}")
    
    result = gradcam.visualize(
        sample_image,
        save_path=config.RESULTS_DIR / "gradcam_single.png"
    )
    
    print(f"\nTahmin: {result['predicted_class']}")
    print(f"Güven: {result['confidence']:.2f}%")
    
    # Batch Grad-CAM
    print("\n" + "=" * 60)
    print("BATCH GRAD-CAM")
    print("=" * 60)
    
    sample_images = np.random.choice(image_files, size=9, replace=False)
    gradcam.batch_visualize(sample_images)
    
    print("\n✅ Grad-CAM görselleştirmeleri tamamlandı!")
    print(f"📁 Sonuçlar: {config.RESULTS_DIR}")
