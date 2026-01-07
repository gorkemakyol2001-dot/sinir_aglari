# ==============================
# UYDU GÖRÜNTÜSİ SINIFLANDIRICI - WEB ARAYÜZÜ
# Gradio ile Etkileşimli Demo
# ==============================

import tensorflow as tf
import numpy as np
import gradio as gr
from PIL import Image
import os

# ==============================
# MODEL YÜKLEME
# ==============================

MODEL_PATH = "outputs/satellite_model.keras"

# Model kontrolü
if not os.path.exists(MODEL_PATH):
    print("❌ Model bulunamadı! Önce main_improved.py'yi çalıştırın.")
    exit(1)

print("📦 Model yükleniyor...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model başarıyla yüklendi!")

# Sınıf isimleri (EuroSAT veri seti)
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

# Türkçe karşılıklar
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
# TAHMİN FONKSİYONU
# ==============================

def classify_image(image):
    """
    Uydu görüntüsünü sınıflandırır
    
    Args:
        image: PIL Image veya numpy array
        
    Returns:
        dict: Sınıf isimleri ve olasılıkları
    """
    
    if image is None:
        return {"Hata": 1.0}
    
    try:
        # Görüntüyü hazırla
        if isinstance(image, np.ndarray):
            img = Image.fromarray(image.astype('uint8'))
        else:
            img = image
            
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Tahmin yap
        predictions = model.predict(img_array, verbose=0)
        
        # Sonuçları hazırla
        results = {}
        for i, class_name in enumerate(CLASS_NAMES):
            turkish_name = CLASS_NAMES_TR.get(class_name, class_name)
            results[turkish_name] = float(predictions[0][i])
        
        # En yüksek 5 tahmini sırala
        sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True)[:5])
        
        return sorted_results
        
    except Exception as e:
        return {"Hata": f"Tahmin yapılırken hata oluştu: {str(e)}"}

# ==============================
# ÖRNEK GÖRÜNTÜLER
# ==============================

def get_example_images():
    """EuroSAT klasöründen örnek görüntüler al"""
    examples = []
    data_dir = r"C:\Users\Lenovo\Desktop\sinir ağları\EuroSAT"
    
    if os.path.exists(data_dir):
        for class_name in CLASS_NAMES[:5]:  # İlk 5 sınıftan örnek
            class_path = os.path.join(data_dir, class_name)
            if os.path.isdir(class_path):
                images = os.listdir(class_path)
                if images:
                    example_path = os.path.join(class_path, images[0])
                    examples.append([example_path])
    
    return examples if examples else None

# ==============================
# GRADIO ARAYÜZÜ
# ==============================

# CSS ile özel stil
custom_css = """
#title {
    text-align: center;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5em;
    font-weight: bold;
    margin-bottom: 10px;
}

#description {
    text-align: center;
    font-size: 1.1em;
    color: #666;
    margin-bottom: 20px;
}

.gradio-container {
    max-width: 900px;
    margin: auto;
}
"""

# Arayüz oluştur
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    
    gr.HTML("<h1 id='title'>🛰️ Uydu Görüntüsü Arazi Sınıflandırıcı</h1>")
    gr.HTML("<p id='description'>EuroSAT veri seti ile eğitilmiş derin öğrenme modeli kullanarak uydu görüntülerini sınıflandırır</p>")
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="📸 Uydu Görüntüsü Yükleyin",
                type="pil",
                height=400
            )
            
            classify_btn = gr.Button(
                "🔍 Sınıflandır",
                variant="primary",
                size="lg"
            )
            
            gr.Markdown("### 📋 Sınıflandırılabilir Arazi Tipleri:")
            gr.Markdown("""
            - 🌾 Yıllık Ekin
            - 🌲 Orman
            - 🌿 Otsu Bitki Örtüsü
            - 🛣️ Otoyol
            - 🏭 Sanayi Bölgesi
            - 🐄 Mera
            - 🌳 Kalıcı Ekin
            - 🏘️ Yerleşim Alanı
            - 🌊 Nehir
            - 💧 Deniz/Göl
            """)
        
        with gr.Column(scale=1):
            label_output = gr.Label(
                label="📊 Tahmin Sonuçları",
                num_top_classes=5
            )
            
            gr.Markdown("### ℹ️ Nasıl Kullanılır?")
            gr.Markdown("""
            1. Sol taraftan bir uydu görüntüsü yükleyin
            2. "Sınıflandır" butonuna tıklayın
            3. Model, görüntüyü analiz edip en olası 5 arazi tipini gösterecektir
            4. Yüzde değerleri, modelin tahmin güvenini gösterir
            """)
            
            gr.Markdown("### 🎯 Model Bilgileri")
            gr.Markdown("""
            - **Mimari**: MobileNetV2 (Transfer Learning)
            - **Veri Seti**: EuroSAT
            - **Görüntü Boyutu**: 224x224
            - **Sınıf Sayısı**: 10
            """)
    
    # Örnek görüntüler
    examples = get_example_images()
    if examples:
        gr.Examples(
            examples=examples,
            inputs=image_input,
            outputs=label_output,
            fn=classify_image,
            cache_examples=False,
            label="🖼️ Örnek Görüntüler"
        )
    
    # Buton tıklama olayı
    classify_btn.click(
        fn=classify_image,
        inputs=image_input,
        outputs=label_output
    )
    
    # Footer
    gr.HTML("""
    <div style='text-align: center; margin-top: 30px; padding: 20px; background-color: #f5f5f5; border-radius: 10px;'>
        <p style='color: #666; margin: 0;'>
            🚀 MobileNetV2 ile Transfer Learning kullanılarak geliştirilmiştir<br>
            📚 EuroSAT Veri Seti | 🤖 TensorFlow & Keras
        </p>
    </div>
    """)

# ==============================
# UYGULAMAYI BAŞLAT
# ==============================

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("🚀 WEB ARAYÜZÜ BAŞLATILIYOR...")
    print("=" * 50)
    
    demo.launch(
        share=False,  # True yaparsanız public link alırsınız
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        quiet=False
    )
