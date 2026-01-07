# ==============================
# MULTI-MODEL TRAINER
# Çoklu Model Mimarisi Eğitimi ve Karşılaştırma
# ==============================

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import json
import config
import utils

logger = utils.setup_logging(__name__)

# ==============================
# MODEL FACTORY
# ==============================

class ModelFactory:
    """Farklı model mimarilerini oluştur"""
    
    @staticmethod
    def create_model(
        model_name: str,
        num_classes: int,
        input_shape: tuple = None
    ) -> tuple:
        """
        Model oluştur
        
        Args:
            model_name: Model adı ('mobilenetv2', 'efficientnetb0', etc.)
            num_classes: Sınıf sayısı
            input_shape: Girdi boyutu
        
        Returns:
            (model, base_model) tuple
        """
        if input_shape is None:
            input_shape = config.AVAILABLE_MODELS[model_name]['input_size'] + (3,)
        
        logger.info(f"Model oluşturuluyor: {model_name}")
        
        if model_name == 'mobilenetv2':
            base_model = tf.keras.applications.MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=input_shape
            )
        
        elif model_name == 'efficientnetb0':
            base_model = tf.keras.applications.EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=input_shape
            )
        
        elif model_name == 'efficientnetb3':
            base_model = tf.keras.applications.EfficientNetB3(
                weights='imagenet',
                include_top=False,
                input_shape=input_shape
            )
        
        elif model_name == 'resnet50':
            base_model = tf.keras.applications.ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=input_shape
            )
        
        else:
            raise ValueError(f"Desteklenmeyen model: {model_name}")
        
        # Base model'i dondur
        base_model.trainable = False
        
        # Sınıflandırma katmanları
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        
        model = tf.keras.models.Model(inputs=base_model.input, outputs=output)
        
        return model, base_model

# ==============================
# MULTI-MODEL TRAINER
# ==============================

class MultiModelTrainer:
    """Birden fazla model mimarisini eğit ve karşılaştır"""
    
    def __init__(
        self,
        model_names: list = None,
        data_dir: Path = config.DATA_DIR,
        save_dir: Path = config.MODELS_DIR
    ):
        """
        Args:
            model_names: Eğitilecek model isimleri
            data_dir: Veri dizini
            save_dir: Model kayıt dizini
        """
        self.model_names = model_names or ['mobilenetv2', 'efficientnetb0', 'resnet50']
        self.data_dir = Path(data_dir)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        self.models = {}
        self.histories = {}
        self.results = {}
        
        logger.info(f"Eğitilecek modeller: {self.model_names}")
    
    def prepare_data(self, model_name: str):
        """Veri setini hazırla"""
        img_size = config.AVAILABLE_MODELS[model_name]['input_size']
        
        # Data generators
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            validation_split=config.VALIDATION_SPLIT,
            **config.AUGMENTATION_CONFIG
        )
        
        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            validation_split=config.VALIDATION_SPLIT
        )
        
        train_data = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=img_size,
            batch_size=config.BATCH_SIZE,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        val_data = val_datagen.flow_from_directory(
            self.data_dir,
            target_size=img_size,
            batch_size=config.BATCH_SIZE,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        return train_data, val_data
    
    def train_model(
        self,
        model_name: str,
        epochs: int = config.EPOCHS,
        verbose: int = 1
    ):
        """Tek bir modeli eğit"""
        logger.info(f"\n{'='*60}")
        logger.info(f"MODEL EĞİTİMİ: {model_name.upper()}")
        logger.info(f"{'='*60}")
        
        # Veri hazırla
        train_data, val_data = self.prepare_data(model_name)
        num_classes = train_data.num_classes
        
        # Model oluştur
        model, base_model = ModelFactory.create_model(model_name, num_classes)
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                **config.CALLBACKS_CONFIG['early_stopping'],
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                **config.CALLBACKS_CONFIG['reduce_lr'],
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                str(self.save_dir / f"{model_name}_best.keras"),
                **config.CALLBACKS_CONFIG['model_checkpoint'],
                verbose=1
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=str(config.LOGS_DIR / f"fit/{model_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"),
                histogram_freq=1
            )
        ]
        
        # Model derle
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
        )
        
        # Eğitim
        logger.info(f"\n🚀 Eğitim başlıyor (1. Aşama - Frozen Base)...")
        history1 = model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose
        )
        
        # Fine-tuning
        logger.info(f"\n🔧 Fine-tuning başlıyor (2. Aşama)...")
        base_model.trainable = True
        
        # İlk katmanları dondur
        for layer in base_model.layers[:int(len(base_model.layers) * 0.7)]:
            layer.trainable = False
        
        # Yeniden derle
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE / 10),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
        )
        
        # Fine-tuning eğitimi
        history2 = model.fit(
            train_data,
            validation_data=val_data,
            epochs=10,
            callbacks=callbacks,
            verbose=verbose
        )
        
        # Geçmişi birleştir
        combined_history = {
            'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
            'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
            'loss': history1.history['loss'] + history2.history['loss'],
            'val_loss': history1.history['val_loss'] + history2.history['val_loss']
        }
        
        # Kaydet
        self.models[model_name] = model
        self.histories[model_name] = combined_history
        
        # Değerlendirme
        logger.info(f"\n📊 Model değerlendiriliyor...")
        val_loss, val_acc, val_top3 = model.evaluate(val_data, verbose=0)
        
        self.results[model_name] = {
            'val_loss': float(val_loss),
            'val_accuracy': float(val_acc),
            'val_top3_accuracy': float(val_top3),
            'total_epochs': len(combined_history['accuracy']),
            'best_val_accuracy': float(max(combined_history['val_accuracy'])),
            'model_params': model.count_params()
        }
        
        logger.info(f"\n✅ {model_name} eğitimi tamamlandı!")
        logger.info(f"Validation Accuracy: {val_acc:.4f}")
        logger.info(f"Validation Top-3 Accuracy: {val_top3:.4f}")
        
        return model, combined_history
    
    def train_all(self, epochs: int = config.EPOCHS):
        """Tüm modelleri eğit"""
        logger.info(f"\n{'='*60}")
        logger.info(f"TÜM MODELLER EĞİTİLİYOR")
        logger.info(f"{'='*60}")
        logger.info(f"Modeller: {', '.join(self.model_names)}")
        logger.info(f"Epoch: {epochs}")
        
        for model_name in self.model_names:
            try:
                self.train_model(model_name, epochs=epochs)
            except Exception as e:
                logger.error(f"❌ {model_name} eğitimi başarısız: {e}")
                continue
        
        # Sonuçları kaydet
        self.save_results()
        
        # Karşılaştırma
        self.compare_models()
    
    def save_results(self):
        """Sonuçları kaydet"""
        results_file = self.save_dir / "model_comparison_results.json"
        utils.save_json(self.results, results_file)
        logger.info(f"📁 Sonuçlar kaydedildi: {results_file}")
    
    def compare_models(self):
        """Modelleri karşılaştır"""
        if not self.results:
            logger.warning("Karşılaştırılacak sonuç yok!")
            return
        
        logger.info(f"\n{'='*60}")
        logger.info("MODEL KARŞILAŞTIRMA")
        logger.info(f"{'='*60}")
        
        # Tablo oluştur
        print(f"\n{'Model':<20} {'Val Acc':<12} {'Top-3 Acc':<12} {'Params':<15}")
        print("-" * 60)
        
        for model_name, results in self.results.items():
            print(f"{model_name:<20} "
                  f"{results['val_accuracy']:<12.4f} "
                  f"{results['val_top3_accuracy']:<12.4f} "
                  f"{results['model_params']:<15,}")
        
        # Görselleştirme
        self.plot_comparison()
    
    def plot_comparison(self):
        """Karşılaştırma grafikleri"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Validation Accuracy
        ax = axes[0, 0]
        models = list(self.results.keys())
        val_accs = [self.results[m]['val_accuracy'] for m in models]
        
        bars = ax.bar(models, val_accs, color='skyblue', edgecolor='navy')
        ax.set_ylabel('Validation Accuracy', fontsize=12)
        ax.set_title('Model Validation Accuracy Karşılaştırması', fontsize=14, fontweight='bold')
        ax.set_ylim([min(val_accs) - 0.05, 1.0])
        
        # Değerleri göster
        for bar, val in zip(bars, val_accs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=10)
        
        # 2. Model Parameters
        ax = axes[0, 1]
        params = [self.results[m]['model_params'] / 1e6 for m in models]  # Millions
        
        bars = ax.bar(models, params, color='lightcoral', edgecolor='darkred')
        ax.set_ylabel('Parameters (Millions)', fontsize=12)
        ax.set_title('Model Parametre Sayısı', fontsize=14, fontweight='bold')
        
        for bar, val in zip(bars, params):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}M', ha='center', va='bottom', fontsize=10)
        
        # 3. Training History
        ax = axes[1, 0]
        for model_name, history in self.histories.items():
            ax.plot(history['val_accuracy'], label=model_name, linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Validation Accuracy', fontsize=12)
        ax.set_title('Eğitim Geçmişi Karşılaştırması', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Top-3 Accuracy
        ax = axes[1, 1]
        top3_accs = [self.results[m]['val_top3_accuracy'] for m in models]
        
        bars = ax.bar(models, top3_accs, color='lightgreen', edgecolor='darkgreen')
        ax.set_ylabel('Top-3 Accuracy', fontsize=12)
        ax.set_title('Model Top-3 Accuracy Karşılaştırması', fontsize=14, fontweight='bold')
        ax.set_ylim([min(top3_accs) - 0.05, 1.0])
        
        for bar, val in zip(bars, top3_accs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        save_path = config.RESULTS_DIR / "model_comparison.png"
        plt.savefig(save_path, dpi=300)
        logger.info(f"📊 Karşılaştırma grafiği kaydedildi: {save_path}")
        
        plt.show()

# ==============================
# MAIN EXECUTION
# ==============================

if __name__ == "__main__":
    print("=" * 60)
    print("ÇOK MODELLİ EĞİTİM SİSTEMİ")
    print("=" * 60)
    
    # Trainer oluştur
    trainer = MultiModelTrainer(
        model_names=['mobilenetv2', 'efficientnetb0', 'resnet50']
    )
    
    # Tüm modelleri eğit
    trainer.train_all(epochs=15)
    
    print("\n✅ Tüm modeller eğitildi ve karşılaştırıldı!")
    print(f"📁 Modeller: {config.MODELS_DIR}")
    print(f"📁 Sonuçlar: {config.RESULTS_DIR}")
