# ==============================
# MODEL EXPORT
# Modeli Farklı Formatlara Dışa Aktar
# ==============================

import tensorflow as tf
import numpy as np
from pathlib import Path
from typing import Union
import config
import utils

logger = utils.setup_logging(__name__)

# ==============================
# MODEL EXPORTER
# ==============================

class ModelExporter:
    """Model dışa aktarma aracı"""
    
    def __init__(self, model_path: Union[str, Path]):
        """
        Args:
            model_path: Model dosya yolu
        """
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model bulunamadı: {model_path}")
        
        logger.info(f"Model yükleniyor: {model_path}")
        self.model = utils.load_model_safe(model_path)
        
        self.export_dir = config.MODELS_DIR / "exported"
        self.export_dir.mkdir(exist_ok=True, parents=True)
    
    def export_h5(self, output_path: Union[str, Path] = None):
        """HDF5 formatına aktar"""
        if output_path is None:
            output_path = self.export_dir / f"{self.model_path.stem}.h5"
        
        logger.info(f"H5 formatına aktarılıyor: {output_path}")
        
        try:
            self.model.save(output_path, save_format='h5')
            logger.info(f"✅ H5 export başarılı: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"❌ H5 export hatası: {e}")
            raise
    
    def export_savedmodel(self, output_path: Union[str, Path] = None):
        """TensorFlow SavedModel formatına aktar"""
        if output_path is None:
            output_path = self.export_dir / f"{self.model_path.stem}_savedmodel"
        
        logger.info(f"SavedModel formatına aktarılıyor: {output_path}")
        
        try:
            tf.saved_model.save(self.model, str(output_path))
            logger.info(f"✅ SavedModel export başarılı: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"❌ SavedModel export hatası: {e}")
            raise
    
    def export_tflite(
        self,
        output_path: Union[str, Path] = None,
        quantize: bool = True
    ):
        """
        TensorFlow Lite formatına aktar
        
        Args:
            output_path: Çıktı yolu
            quantize: Quantization uygula (model boyutunu küçültür)
        """
        if output_path is None:
            suffix = "_quantized" if quantize else ""
            output_path = self.export_dir / f"{self.model_path.stem}{suffix}.tflite"
        
        logger.info(f"TFLite formatına aktarılıyor: {output_path}")
        logger.info(f"Quantization: {'Evet' if quantize else 'Hayır'}")
        
        try:
            # Converter oluştur
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            
            if quantize:
                # Dynamic range quantization
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                
                # Float16 quantization (daha iyi performans)
                converter.target_spec.supported_types = [tf.float16]
            
            # Convert
            tflite_model = converter.convert()
            
            # Kaydet
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            # Boyut bilgisi
            original_size = self.model_path.stat().st_size / (1024 * 1024)  # MB
            tflite_size = len(tflite_model) / (1024 * 1024)  # MB
            
            logger.info(f"✅ TFLite export başarılı: {output_path}")
            logger.info(f"Orijinal boyut: {original_size:.2f} MB")
            logger.info(f"TFLite boyut: {tflite_size:.2f} MB")
            logger.info(f"Sıkıştırma: {(1 - tflite_size/original_size)*100:.1f}%")
            
            return output_path
        
        except Exception as e:
            logger.error(f"❌ TFLite export hatası: {e}")
            raise
    
    def export_onnx(self, output_path: Union[str, Path] = None):
        """
        ONNX formatına aktar
        
        Not: tf2onnx kütüphanesi gereklidir
        """
        if output_path is None:
            output_path = self.export_dir / f"{self.model_path.stem}.onnx"
        
        logger.info(f"ONNX formatına aktarılıyor: {output_path}")
        
        try:
            import tf2onnx
            
            # Model spec
            spec = (tf.TensorSpec(self.model.input_shape, tf.float32, name="input"),)
            
            # Convert
            model_proto, _ = tf2onnx.convert.from_keras(
                self.model,
                input_signature=spec,
                opset=13
            )
            
            # Kaydet
            with open(output_path, 'wb') as f:
                f.write(model_proto.SerializeToString())
            
            logger.info(f"✅ ONNX export başarılı: {output_path}")
            return output_path
        
        except ImportError:
            logger.error("❌ tf2onnx kütüphanesi yüklü değil!")
            logger.info("Yüklemek için: pip install tf2onnx")
            raise
        
        except Exception as e:
            logger.error(f"❌ ONNX export hatası: {e}")
            raise
    
    def export_all(self, quantize_tflite: bool = True):
        """Tüm formatlara aktar"""
        logger.info(f"\n{'='*60}")
        logger.info("TÜM FORMATLARA AKTARILIYOR")
        logger.info(f"{'='*60}")
        
        results = {}
        
        # H5
        try:
            results['h5'] = self.export_h5()
        except Exception as e:
            results['h5'] = f"Hata: {e}"
        
        # SavedModel
        try:
            results['savedmodel'] = self.export_savedmodel()
        except Exception as e:
            results['savedmodel'] = f"Hata: {e}"
        
        # TFLite
        try:
            results['tflite'] = self.export_tflite(quantize=quantize_tflite)
        except Exception as e:
            results['tflite'] = f"Hata: {e}"
        
        # ONNX
        try:
            results['onnx'] = self.export_onnx()
        except Exception as e:
            results['onnx'] = f"Hata: {e}"
        
        # Özet
        logger.info(f"\n{'='*60}")
        logger.info("EXPORT ÖZET")
        logger.info(f"{'='*60}")
        
        for format_name, result in results.items():
            if isinstance(result, Path):
                logger.info(f"✅ {format_name.upper()}: {result}")
            else:
                logger.info(f"❌ {format_name.upper()}: {result}")
        
        return results

# ==============================
# TFLITE INFERENCE TEST
# ==============================

def test_tflite_model(
    tflite_path: Union[str, Path],
    test_image_path: Union[str, Path]
):
    """TFLite modelini test et"""
    logger.info(f"\n{'='*60}")
    logger.info("TFLITE MODEL TESTİ")
    logger.info(f"{'='*60}")
    
    # TFLite interpreter
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    
    # Input/output detayları
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    logger.info(f"Input shape: {input_details[0]['shape']}")
    logger.info(f"Output shape: {output_details[0]['shape']}")
    
    # Test görüntüsü
    img_array = utils.load_and_preprocess_image(test_image_path)
    
    # Tahmin
    interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))
    interpreter.invoke()
    
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # Sonuç
    top_idx = np.argmax(predictions)
    top_class = config.CLASS_NAMES[top_idx]
    confidence = predictions[top_idx]
    
    logger.info(f"\nTahmin: {top_class}")
    logger.info(f"Güven: {confidence:.4f}")
    
    return top_class, confidence

# ==============================
# MAIN EXECUTION
# ==============================

if __name__ == "__main__":
    print("=" * 60)
    print("MODEL EXPORT ARACI")
    print("=" * 60)
    
    # Model yolu
    model_path = config.MODELS_DIR / "eurosat_best_model.keras"
    
    if not model_path.exists():
        print(f"\n❌ Model bulunamadı: {model_path}")
        print("Önce modeli eğitin!")
        exit(1)
    
    # Exporter oluştur
    exporter = ModelExporter(model_path)
    
    # Tüm formatlara aktar
    results = exporter.export_all(quantize_tflite=True)
    
    # TFLite test
    if 'tflite' in results and isinstance(results['tflite'], Path):
        print("\n" + "=" * 60)
        print("TFLITE MODEL TEST EDİLİYOR")
        print("=" * 60)
        
        # Test görüntüsü
        image_files = utils.get_image_files(config.DATA_DIR)
        if image_files:
            test_image = np.random.choice(image_files)
            print(f"Test görüntüsü: {test_image}")
            
            try:
                test_tflite_model(results['tflite'], test_image)
            except Exception as e:
                print(f"❌ Test hatası: {e}")
    
    print("\n✅ Model export tamamlandı!")
    print(f"📁 Export dizini: {exporter.export_dir}")
