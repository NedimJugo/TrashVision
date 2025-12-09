from ultralytics import YOLO
from PIL import Image
import sys

def predict_image(image_path):
    # Učitaj model
    model = YOLO("models/trashvision_v1/weights/best.pt")
    
    # Klase
    classes = ['battery', 'biological', 'cardboard', 'clothes', 'glass', 
               'metal', 'paper', 'plastic', 'shoes', 'trash']
    
    # Predikcija
    results = model(image_path)[0]
    
    # Top 3 predikcije
    probs = results.probs
    top3_indices = probs.top5[:3]
    top3_probs = probs.data[top3_indices].cpu().numpy()
    
    print("\n" + "=" * 50)
    print(f"📸 Slika: {image_path}")
    print("=" * 50)
    print("\n🎯 Top 3 predikcije:\n")
    
    for i, (idx, prob) in enumerate(zip(top3_indices, top3_probs), 1):
        print(f"{i}. {classes[idx]:12s} - {prob*100:5.1f}% confidence")
    
    print("\n✅ Predviđena klasa: " + classes[probs.top1])
    print("=" * 50)
    
    # Prikaži sliku
    img = Image.open(image_path)
    img.show()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Upotreba: python src/predict.py <putanja_do_slike>")
        print("\nPrimjer:")
        print("  python src/predict.py data/processed/test/plastic/plastic_100.jpg")
    else:
        predict_image(sys.argv[1])