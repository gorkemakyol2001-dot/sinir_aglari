# ==============================
# REST API SERVER
# FastAPI ile Uydu Görüntüsü Sınıflandırma API
# ==============================

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from pathlib import Path
import uvicorn
import config
import utils
from ensemble_predictor import EnsemblePredictor

logger = utils.setup_logging(__name__)

# ==============================
# FASTAPI APP
# ==============================

app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description=config.API_DESCRIPTION
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# GLOBAL VARIABLES
# ==============================

models = {}
ensemble = None

# ==============================
# PYDANTIC MODELS
# ==============================

class PredictionResponse(BaseModel):
    """Tahmin yanıt modeli"""
    success: bool
    predicted_class: str
    confidence: float
    top5_predictions: dict
    model_used: str

class BatchPredictionResponse(BaseModel):
    """Toplu tahmin yanıt modeli"""
    success: bool
    total_images: int
    predictions: List[dict]

class HealthResponse(BaseModel):
    """Sağlık kontrolü yanıt modeli"""
    status: str
    models_loaded: List[str]
    ensemble_available: bool

# ==============================
# STARTUP & SHUTDOWN
# ==============================

@app.on_event("startup")
async def startup_event():
    """Uygulama başlangıcında modelleri yükle"""
    global models, ensemble
    
    logger.info("=" * 60)
    logger.info("API SUNUCUSU BAŞLATILIYOR")
    logger.info("=" * 60)
    
    # Modelleri yükle
    logger.info("Modeller yükleniyor...")
    
    for model_name in ['mobilenetv2', 'efficientnetb0', 'resnet50']:
        model_path = config.MODELS_DIR / f"{model_name}_best.keras"
        
        if model_path.exists():
            try:
                models[model_name] = utils.load_model_safe(model_path)
                logger.info(f"✅ {model_name} yüklendi")
            except Exception as e:
                logger.warning(f"⚠️ {model_name} yüklenemedi: {e}")
    
    # Ensemble oluştur
    if len(models) > 1:
        try:
            ensemble = EnsemblePredictor(
                model_paths={name: None for name in models.keys()},
                weights=config.ENSEMBLE_WEIGHTS,
                strategy='weighted_average'
            )
            ensemble.models = models  # Zaten yüklenmiş modelleri kullan
            logger.info("✅ Ensemble predictor hazır")
        except Exception as e:
            logger.warning(f"⚠️ Ensemble oluşturulamadı: {e}")
    
    if not models:
        logger.error("❌ Hiç model yüklenemedi!")
    else:
        logger.info(f"✅ {len(models)} model yüklendi")
        logger.info("🚀 API sunucusu hazır!")

@app.on_event("shutdown")
async def shutdown_event():
    """Uygulama kapanışında temizlik"""
    logger.info("API sunucusu kapatılıyor...")

# ==============================
# HELPER FUNCTIONS
# ==============================

def process_image(file: UploadFile) -> np.ndarray:
    """Yüklenen dosyayı işle"""
    try:
        # Dosyayı oku
        contents = file.file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Resize
        image = image.resize(config.IMG_SIZE)
        
        # Array'e çevir ve normalize
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Görüntü işleme hatası: {str(e)}")
    
    finally:
        file.file.close()

# ==============================
# API ENDPOINTS
# ==============================

@app.get("/", response_model=dict)
async def root():
    """Ana sayfa"""
    return {
        "message": "🛰️ Satellite Image Classifier API",
        "version": config.API_VERSION,
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Sağlık kontrolü"""
    return HealthResponse(
        status="healthy" if models else "unhealthy",
        models_loaded=list(models.keys()),
        ensemble_available=ensemble is not None
    )

@app.get("/models", response_model=dict)
async def list_models():
    """Mevcut modelleri listele"""
    model_info = {}
    
    for name, model in models.items():
        model_info[name] = utils.get_model_info(model)
    
    return {
        "available_models": list(models.keys()),
        "model_details": model_info,
        "ensemble_available": ensemble is not None
    }

@app.get("/classes", response_model=dict)
async def list_classes():
    """Sınıfları listele"""
    return {
        "classes": config.CLASS_NAMES,
        "classes_turkish": config.CLASS_NAMES_TR,
        "total_classes": len(config.CLASS_NAMES)
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    model_name: Optional[str] = Query(None, description="Model adı (None ise ensemble)")
):
    """
    Tek görüntü tahmini
    
    Args:
        file: Görüntü dosyası
        model_name: Kullanılacak model (None ise ensemble)
    
    Returns:
        Tahmin sonuçları
    """
    try:
        # Görüntüyü işle
        img_array = process_image(file)
        
        # Model seç
        if model_name and model_name in models:
            # Belirli bir model
            model = models[model_name]
            predictions = model.predict(img_array, verbose=0)[0]
            model_used = model_name
        
        elif ensemble:
            # Ensemble
            # Geçici dosya oluştur
            temp_path = config.RESULTS_DIR / "temp_upload.jpg"
            Image.fromarray((img_array[0] * 255).astype(np.uint8)).save(temp_path)
            
            result = ensemble.predict(temp_path)
            
            # Temp dosyayı sil
            temp_path.unlink()
            
            return PredictionResponse(
                success=True,
                predicted_class=result['ensemble_prediction']['class'],
                confidence=result['ensemble_prediction']['confidence'],
                top5_predictions=result['ensemble_prediction']['top5'],
                model_used="ensemble"
            )
        
        else:
            # İlk mevcut model
            model_used = list(models.keys())[0]
            model = models[model_used]
            predictions = model.predict(img_array, verbose=0)[0]
        
        # Sonuçları hazırla
        top_idx = np.argmax(predictions)
        top_class = config.CLASS_NAMES[top_idx]
        confidence = float(predictions[top_idx])
        
        # Top-5
        top5_indices = np.argsort(predictions)[-5:][::-1]
        top5_predictions = {
            config.CLASS_NAMES[i]: float(predictions[i])
            for i in top5_indices
        }
        
        return PredictionResponse(
            success=True,
            predicted_class=top_class,
            confidence=confidence,
            top5_predictions=top5_predictions,
            model_used=model_used
        )
    
    except Exception as e:
        logger.error(f"Tahmin hatası: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_predict")
async def batch_predict(files: List[UploadFile] = File(...)):
    """
    Toplu tahmin
    
    Args:
        files: Görüntü dosyaları listesi
    
    Returns:
        Toplu tahmin sonuçları
    """
    try:
        results = []
        
        for file in files:
            try:
                img_array = process_image(file)
                
                # İlk mevcut modeli kullan
                model_name = list(models.keys())[0]
                model = models[model_name]
                
                predictions = model.predict(img_array, verbose=0)[0]
                top_idx = np.argmax(predictions)
                
                results.append({
                    "filename": file.filename,
                    "predicted_class": config.CLASS_NAMES[top_idx],
                    "confidence": float(predictions[top_idx]),
                    "success": True
                })
            
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "error": str(e),
                    "success": False
                })
        
        return BatchPredictionResponse(
            success=True,
            total_images=len(files),
            predictions=results
        )
    
    except Exception as e:
        logger.error(f"Batch tahmin hatası: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats", response_model=dict)
async def get_stats():
    """API istatistikleri"""
    return {
        "total_models": len(models),
        "total_classes": len(config.CLASS_NAMES),
        "ensemble_enabled": ensemble is not None,
        "api_version": config.API_VERSION
    }

# ==============================
# MAIN EXECUTION
# ==============================

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 REST API SUNUCUSU BAŞLATILIYOR")
    print("=" * 60)
    print(f"\nAPI Adresi: http://{config.API_HOST}:{config.API_PORT}")
    print(f"Dokümantasyon: http://{config.API_HOST}:{config.API_PORT}/docs")
    print(f"Alternatif Docs: http://{config.API_HOST}:{config.API_PORT}/redoc")
    print("\n" + "=" * 60)
    
    uvicorn.run(
        app,
        host=config.API_HOST,
        port=config.API_PORT,
        log_level="info"
    )
