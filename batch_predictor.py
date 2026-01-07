# ==============================
# BATCH PREDICTOR
# Toplu Görüntü Tahmini Aracı
# ==============================

import argparse
from pathlib import Path
import pandas as pd
import json
from typing import Union, List
import config
import utils
from ensemble_predictor import EnsemblePredictor

logger = utils.setup_logging(__name__)

# ==============================
# BATCH PREDICTOR
# ==============================

class BatchPredictor:
    """Toplu tahmin aracı"""
    
    def __init__(
        self,
        model_path: Union[str, Path] = None,
        use_ensemble: bool = False
    ):
        """
        Args:
            model_path: Model dosya yolu (None ise varsayılan)
            use_ensemble: Ensemble kullan
        """
        self.use_ensemble = use_ensemble
        
        if use_ensemble:
            logger.info("Ensemble predictor kullanılıyor...")
            self.predictor = EnsemblePredictor()
        else:
            # Tek model
            if model_path is None:
                model_path = config.MODELS_DIR / "eurosat_best_model.keras"
            
            logger.info(f"Model yükleniyor: {model_path}")
            self.model = utils.load_model_safe(model_path)
            self.predictor = None
    
    def predict_directory(
        self,
        input_dir: Union[str, Path],
        output_file: Union[str, Path] = None,
        output_format: str = 'csv',
        recursive: bool = True
    ):
        """
        Bir dizindeki tüm görüntüleri tahmin et
        
        Args:
            input_dir: Girdi dizini
            output_file: Çıktı dosyası
            output_format: Çıktı formatı ('csv', 'json')
            recursive: Alt dizinleri tara
        """
        input_dir = Path(input_dir)
        
        if not input_dir.exists():
            raise ValueError(f"Dizin bulunamadı: {input_dir}")
        
        # Görüntüleri bul
        logger.info(f"Görüntüler aranıyor: {input_dir}")
        
        if recursive:
            image_files = utils.get_image_files(input_dir)
        else:
            image_files = [
                f for f in input_dir.iterdir()
                if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
            ]
        
        logger.info(f"✅ {len(image_files)} görüntü bulundu")
        
        if not image_files:
            logger.warning("Hiç görüntü bulunamadı!")
            return
        
        # Tahminler
        logger.info("Tahminler yapılıyor...")
        results = []
        
        pbar = utils.create_progress_bar(len(image_files), "Tahmin")
        
        for img_path in image_files:
            try:
                if self.use_ensemble:
                    # Ensemble tahmin
                    result = self.predictor.predict(img_path)
                    pred_class = result['ensemble_prediction']['class']
                    confidence = result['ensemble_prediction']['confidence']
                else:
                    # Tek model tahmin
                    result = utils.predict_single_image(
                        self.model,
                        img_path,
                        config.CLASS_NAMES
                    )
                    pred_class = result['top_class']
                    confidence = result['confidence']
                
                results.append({
                    'image_path': str(img_path),
                    'filename': img_path.name,
                    'predicted_class': pred_class,
                    'confidence': confidence,
                    'status': 'success'
                })
            
            except Exception as e:
                logger.error(f"Tahmin hatası ({img_path}): {e}")
                results.append({
                    'image_path': str(img_path),
                    'filename': img_path.name,
                    'predicted_class': None,
                    'confidence': None,
                    'status': 'error',
                    'error': str(e)
                })
            
            if pbar:
                pbar.update(1)
        
        if pbar:
            pbar.close()
        
        # Sonuçları kaydet
        if output_file is None:
            output_file = config.RESULTS_DIR / f"batch_predictions.{output_format}"
        
        output_file = Path(output_file)
        
        if output_format == 'csv':
            df = pd.DataFrame(results)
            df.to_csv(output_file, index=False, encoding='utf-8')
            logger.info(f"✅ CSV kaydedildi: {output_file}")
        
        elif output_format == 'json':
            utils.save_json(results, output_file)
            logger.info(f"✅ JSON kaydedildi: {output_file}")
        
        else:
            raise ValueError(f"Desteklenmeyen format: {output_format}")
        
        # Özet
        successful = sum(1 for r in results if r['status'] == 'success')
        failed = len(results) - successful
        
        logger.info(f"\n{'='*60}")
        logger.info(f"ÖZET")
        logger.info(f"{'='*60}")
        logger.info(f"Toplam Görüntü: {len(results)}")
        logger.info(f"Başarılı: {successful}")
        logger.info(f"Başarısız: {failed}")
        logger.info(f"Çıktı: {output_file}")
        
        return results

# ==============================
# COMMAND LINE INTERFACE
# ==============================

def main():
    parser = argparse.ArgumentParser(
        description='Toplu Görüntü Tahmini Aracı'
    )
    
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Girdi dizini'
    )
    
    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help='Çıktı dosyası (varsayılan: results/batch_predictions.csv)'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['csv', 'json'],
        default='csv',
        help='Çıktı formatı'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Model dosya yolu'
    )
    
    parser.add_argument(
        '--ensemble',
        action='store_true',
        help='Ensemble predictor kullan'
    )
    
    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='Alt dizinleri tarama'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TOPLU TAHMİN ARACI")
    print("=" * 60)
    
    # Predictor oluştur
    predictor = BatchPredictor(
        model_path=args.model,
        use_ensemble=args.ensemble
    )
    
    # Tahminleri yap
    predictor.predict_directory(
        input_dir=args.input_dir,
        output_file=args.output_file,
        output_format=args.format,
        recursive=not args.no_recursive
    )
    
    print("\n✅ Toplu tahmin tamamlandı!")

# ==============================
# MAIN EXECUTION
# ==============================

if __name__ == "__main__":
    main()
