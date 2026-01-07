# ==============================
# ENSEMBLE PREDICTOR
# Birden Fazla Modeli Birleştirerek Tahmin
# ==============================

import tensorflow as tf
import numpy as np
from pathlib import Path
from typing import List, Dict, Union
import config
import utils

logger = utils.setup_logging(__name__)

# ==============================
# ENSEMBLE PREDICTOR
# ==============================

class EnsemblePredictor:
    """Ensemble learning ile tahmin"""
    
    def __init__(
        self,
        model_paths: Dict[str, Path] = None,
        weights: Dict[str, float] = None,
        strategy: str = 'weighted_average'
    ):
        """
        Args:
            model_paths: Model dosya yolları {model_name: path}
            weights: Model ağırlıkları {model_name: weight}
            strategy: Ensemble stratejisi ('weighted_average', 'voting', 'max')
        """
        self.strategy = strategy
        self.models = {}
        self.weights = weights or {}
        
        # Modelleri yükle
        if model_paths:
            self.load_models(model_paths)
        else:
            self.auto_load_models()
        
        # Ağırlıkları normalize et
        if self.weights:
            total_weight = sum(self.weights.values())
            self.weights = {k: v/total_weight for k, v in self.weights.items()}
        else:
            # Eşit ağırlık
            n = len(self.models)
            self.weights = {name: 1.0/n for name in self.models.keys()}
        
        logger.info(f"Ensemble Predictor hazır: {len(self.models)} model")
        logger.info(f"Strateji: {self.strategy}")
        logger.info(f"Ağırlıklar: {self.weights}")
    
    def auto_load_models(self):
        """Mevcut modelleri otomatik yükle"""
        logger.info("Modeller otomatik yükleniyor...")
        
        for model_name in config.ENSEMBLE_MODELS:
            model_path = config.MODELS_DIR / f"{model_name}_best.keras"
            
            if model_path.exists():
                try:
                    self.models[model_name] = utils.load_model_safe(model_path)
                    logger.info(f"✅ {model_name} yüklendi")
                except Exception as e:
                    logger.warning(f"⚠️ {model_name} yüklenemedi: {e}")
            else:
                logger.warning(f"⚠️ {model_name} bulunamadı: {model_path}")
        
        if not self.models:
            raise ValueError("Hiç model yüklenemedi! Önce modelleri eğitin.")
    
    def load_models(self, model_paths: Dict[str, Path]):
        """Modelleri yükle"""
        for name, path in model_paths.items():
            try:
                self.models[name] = utils.load_model_safe(path)
                logger.info(f"✅ {name} yüklendi")
            except Exception as e:
                logger.error(f"❌ {name} yüklenemedi: {e}")
    
    def predict(
        self,
        image_path: Union[str, Path],
        return_all: bool = False
    ) -> Dict:
        """
        Ensemble tahmin
        
        Args:
            image_path: Görüntü yolu
            return_all: Tüm model tahminlerini döndür
        
        Returns:
            Tahmin sonuçları
        """
        all_predictions = {}
        
        # Her model için tahmin
        for model_name, model in self.models.items():
            try:
                # Görüntüyü yükle
                preprocess_mode = config.AVAILABLE_MODELS.get(
                    model_name, {}
                ).get('preprocess', 'mobilenet')
                
                img_array = utils.load_and_preprocess_image(
                    image_path,
                    preprocess_mode=preprocess_mode
                )
                
                # Tahmin
                pred = model.predict(img_array, verbose=0)[0]
                all_predictions[model_name] = pred
                
            except Exception as e:
                logger.error(f"❌ {model_name} tahmin hatası: {e}")
                continue
        
        if not all_predictions:
            raise ValueError("Hiç model tahmin yapamadı!")
        
        # Ensemble stratejisi uygula
        if self.strategy == 'weighted_average':
            ensemble_pred = self._weighted_average(all_predictions)
        
        elif self.strategy == 'voting':
            ensemble_pred = self._voting(all_predictions)
        
        elif self.strategy == 'max':
            ensemble_pred = self._max_confidence(all_predictions)
        
        else:
            raise ValueError(f"Bilinmeyen strateji: {self.strategy}")
        
        # Sonuçları hazırla
        top_idx = np.argmax(ensemble_pred)
        top_class = config.CLASS_NAMES[top_idx]
        confidence = float(ensemble_pred[top_idx])
        
        # Top-5 tahminler
        top5_indices = np.argsort(ensemble_pred)[-5:][::-1]
        top5_predictions = {
            config.CLASS_NAMES[i]: float(ensemble_pred[i])
            for i in top5_indices
        }
        
        result = {
            'ensemble_prediction': {
                'class': top_class,
                'confidence': confidence,
                'top5': top5_predictions
            },
            'strategy': self.strategy
        }
        
        # Tüm model tahminlerini ekle
        if return_all:
            individual_predictions = {}
            for model_name, pred in all_predictions.items():
                idx = np.argmax(pred)
                individual_predictions[model_name] = {
                    'class': config.CLASS_NAMES[idx],
                    'confidence': float(pred[idx])
                }
            result['individual_predictions'] = individual_predictions
        
        return result
    
    def _weighted_average(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Ağırlıklı ortalama"""
        ensemble = np.zeros_like(list(predictions.values())[0])
        
        for model_name, pred in predictions.items():
            weight = self.weights.get(model_name, 1.0 / len(predictions))
            ensemble += pred * weight
        
        return ensemble
    
    def _voting(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Çoğunluk oyu (hard voting)"""
        votes = np.zeros(len(config.CLASS_NAMES))
        
        for pred in predictions.values():
            top_class = np.argmax(pred)
            votes[top_class] += 1
        
        # Normalize
        return votes / len(predictions)
    
    def _max_confidence(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """En yüksek güvene sahip tahmin"""
        max_pred = None
        max_conf = -1
        
        for pred in predictions.values():
            conf = np.max(pred)
            if conf > max_conf:
                max_conf = conf
                max_pred = pred
        
        return max_pred
    
    def batch_predict(
        self,
        image_paths: List[Union[str, Path]],
        show_progress: bool = True
    ) -> List[Dict]:
        """Toplu tahmin"""
        results = []
        
        iterator = image_paths
        if show_progress:
            pbar = utils.create_progress_bar(len(image_paths), "Ensemble Prediction")
            if pbar:
                iterator = pbar
        
        for img_path in iterator:
            try:
                result = self.predict(img_path)
                result['image_path'] = str(img_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Tahmin hatası ({img_path}): {e}")
                results.append({
                    'image_path': str(img_path),
                    'error': str(e)
                })
            
            if show_progress and pbar:
                pbar.update(1)
        
        if show_progress and pbar:
            pbar.close()
        
        return results
    
    def compare_strategies(
        self,
        image_path: Union[str, Path]
    ) -> Dict:
        """Farklı stratejileri karşılaştır"""
        strategies = ['weighted_average', 'voting', 'max']
        results = {}
        
        for strategy in strategies:
            original_strategy = self.strategy
            self.strategy = strategy
            
            try:
                result = self.predict(image_path)
                results[strategy] = result['ensemble_prediction']
            except Exception as e:
                logger.error(f"Strateji {strategy} hatası: {e}")
                results[strategy] = {'error': str(e)}
            
            self.strategy = original_strategy
        
        return results

# ==============================
# MAIN EXECUTION
# ==============================

if __name__ == "__main__":
    print("=" * 60)
    print("ENSEMBLE PREDICTOR")
    print("=" * 60)
    
    # Ensemble oluştur
    try:
        ensemble = EnsemblePredictor(
            weights=config.ENSEMBLE_WEIGHTS,
            strategy='weighted_average'
        )
    except Exception as e:
        print(f"\n❌ Hata: {e}")
        print("\nÖnce modelleri eğitin:")
        print("  python multi_model_trainer.py")
        exit(1)
    
    # Örnek görüntü
    print("\n🔍 Örnek görüntü aranıyor...")
    image_files = utils.get_image_files(config.DATA_DIR)
    
    if not image_files:
        print("❌ Görüntü bulunamadı!")
        exit(1)
    
    sample_image = np.random.choice(image_files)
    print(f"Görüntü: {sample_image}")
    
    # Tahmin
    print("\n" + "=" * 60)
    print("ENSEMBLE TAHMİN")
    print("=" * 60)
    
    result = ensemble.predict(sample_image, return_all=True)
    
    print(f"\n🎯 Ensemble Tahmini:")
    print(f"Sınıf: {result['ensemble_prediction']['class']}")
    print(f"Güven: {result['ensemble_prediction']['confidence']:.4f}")
    
    print(f"\n📊 Top-5 Tahminler:")
    for cls, conf in result['ensemble_prediction']['top5'].items():
        print(f"  {cls}: {conf:.4f}")
    
    if 'individual_predictions' in result:
        print(f"\n🤖 Bireysel Model Tahminleri:")
        for model_name, pred in result['individual_predictions'].items():
            print(f"  {model_name}: {pred['class']} ({pred['confidence']:.4f})")
    
    # Strateji karşılaştırması
    print("\n" + "=" * 60)
    print("STRATEJİ KARŞILAŞTIRMASI")
    print("=" * 60)
    
    strategy_results = ensemble.compare_strategies(sample_image)
    
    for strategy, pred in strategy_results.items():
        if 'error' not in pred:
            print(f"\n{strategy}:")
            print(f"  Sınıf: {pred['class']}")
            print(f"  Güven: {pred['confidence']:.4f}")
    
    print("\n✅ Ensemble tahmin tamamlandı!")
