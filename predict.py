# ==============================
# UYDU GÖRÜNTÜSİ TAHMİN SCRIPTI
# Eğitilmiş model ile yeni görüntüleri sınıflandırma
# ==============================

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from tensorflow.keras.preprocessing import image

# ==============================
# SINIF İSİMLERİ
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

# Türkçe açıklamalar
CLASS_DESCRIPTIONS = {
    'AnnualCrop': 'Yıllık Ekin (Buğday, mısır gibi)',
    'Forest': 'Orman',
    'HerbaceousVegetation': 'Otsu Bitki Örtüsü',
    'Highway': 'Otoyol',
    'Industrial': 'Sanayi Bölgesi',
    'Pasture': 'Mera/Otlak',
    'PermanentCrop': 'Kalıcı Ekin (Meyve bahçesi, bağ)',
    'Residential': 'Yerleşim Alanı',
    'River': 'Nehir',
    'SeaLake': 'Deniz/Göl'
}

# ==============================
# TAHMİN FONKSİYONU
# ==============================

def predict_image(model_path, image_path, show_top_n=3):
    """
    Verilen görüntü için tahmin yapar
    
    Args:
        model_path: Eğitilmiş model dosyasının yolu
        image_path: Tahmin yapılacak görüntünün yolu
        show_top_n: Gösterilecek en yüksek N tahmin
    """
    
    # Model yükle
    print(f"\n🔄 Model yükleniyor: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("✅ Model yüklendi!")
    
    # Görüntüyü yükle ve hazırla
    print(f"\n📷 Görüntü yükleniyor: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"❌ HATA: Görüntü bulunamadı: {image_path}")
        return
    
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Batch dimension ekle
    
    # Tahmin yap
    print("\n🔮 Tahmin yapılıyor...")
    predictions = model.predict(img_array, verbose=0)
    
    # En yüksek olasılıklı sınıfları bul
    top_indices = np.argsort(predictions[0])[::-1][:show_top_n]
    
    # Sonuçları yazdır
    print("\n" + "="*70)
    print(f"📊 TAHMİN SONUÇLARI (Top {show_top_n})")
    print("="*70)
    
    for i, idx in enumerate(top_indices, 1):
        class_name = CLASS_NAMES[idx]
        confidence = predictions[0][idx] * 100
        description = CLASS_DESCRIPTIONS[class_name]
        
        print(f"\n{i}. {class_name}")
        print(f"   📝 Açıklama: {description}")
        print(f"   📈 Güven: {confidence:.2f}%")
        print(f"   {'🏆 EN YÜKSEK TAHMİN' if i == 1 else ''}")
    
    print("\n" + "="*70)
    
    # Görselleştirme
    visualize_prediction(img, predictions[0], top_indices)
    
    return CLASS_NAMES[top_indices[0]], predictions[0][top_indices[0]]

# ==============================
# GÖRSELLEŞTİRME
# ==============================

def visualize_prediction(img, predictions, top_indices):
    """
    Tahmin sonuçlarını görselleştirir
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Sol: Görüntü
    ax1.imshow(img)
    ax1.axis('off')
    predicted_class = CLASS_NAMES[top_indices[0]]
    confidence = predictions[top_indices[0]] * 100
    ax1.set_title(f'Tahmin: {predicted_class}\nGüven: {confidence:.1f}%', 
                  fontsize=14, fontweight='bold', color='green')
    
    # Sağ: Olasılık grafiği (top 5)
    top_5_indices = top_indices[:5] if len(top_indices) >= 5 else top_indices
    top_5_probs = [predictions[i] * 100 for i in top_5_indices]
    top_5_names = [CLASS_NAMES[i] for i in top_5_indices]
    
    colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(top_5_names))]
    bars = ax2.barh(top_5_names, top_5_probs, color=colors)
    
    ax2.set_xlabel('Güven (%)', fontsize=12)
    ax2.set_title('Tahmin Olasılıkları (Top 5)', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 100)
    
    # Bar üzerine değerleri yaz
    for bar, prob in zip(bars, top_5_probs):
        width = bar.get_width()
        ax2.text(width + 1, bar.get_y() + bar.get_height()/2, 
                f'{prob:.1f}%', 
                ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

# ==============================
# MAIN
# ==============================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Uydu görüntüsü sınıflandırma tahmini')
    parser.add_argument('--image', type=str, required=True, 
                       help='Tahmin yapılacak görüntünün yolu')
    parser.add_argument('--model', type=str, 
                       default=r'C:\Users\Lenovo\Desktop\sinir ağları\outputs\satellite_model.keras',
                       help='Model dosyasının yolu')
    parser.add_argument('--top', type=int, default=3,
                       help='Gösterilecek en yüksek N tahmin (varsayılan: 3)')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🛰️  UYDU GÖRÜNTÜSİ SINIFLANDIRMA TAHMİNİ")
    print("="*70)
    
    predicted_class, confidence = predict_image(args.model, args.image, args.top)
    
    print(f"\n✅ Tahmin tamamlandı!")
    print(f"🎯 Sonuç: {predicted_class} ({confidence*100:.2f}%)")
    print("="*70 + "\n")
