from ultralytics import YOLO
import torch
from pathlib import Path

def main():
    print("=" * 60)
    print("🚀 TrashVision - YOLOv8 Training")
    print("=" * 60)

    # Provjeri da li je GPU dostupan
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n💻 Device: {device}")

    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
    else:
        print("   ⚠️  GPU nije dostupan, koristim CPU (treniranje će biti sporije)")

    # Učitaj pretreniran YOLOv8 model
    print("\n📦 Učitavam YOLOv8n (nano) model...")
    model = YOLO('yolov8n-cls.pt')  # klasifikacioni model

    # Parametri treniranja
    EPOCHS = 50          # Broj epoha
    BATCH_SIZE = 32      # Batch size
    IMAGE_SIZE = 224     # Veličina slike
    PATIENCE = 10        # Early stopping patience

    print("\n⚙️  Parametri treniranja:")
    print(f"   Epohe: {EPOCHS}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"   Patience: {PATIENCE}")
    print(f"   Device: {device}")

    # Započni treniranje
    print("\n🔥 Započinjem treniranje...\n")
    print("=" * 60)

    try:
        results = model.train(
            data='data/processed',
            epochs=EPOCHS,
            batch=BATCH_SIZE,
            imgsz=IMAGE_SIZE,
            device=device,
            patience=PATIENCE,
            save=True,
            project='models',
            name='trashvision_v1',
            exist_ok=True,
            pretrained=True,
            optimizer='Adam',
            verbose=True,
            seed=42,
            deterministic=True,
            workers=0,  # VAŽNO: Postavi na 0 za Windows
            # Augmentacije
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10,
            translate=0.1,
            scale=0.5,
            fliplr=0.5,
            mosaic=0.0,
        )
        
        print("\n" + "=" * 60)
        print("✅ Treniranje završeno uspješno!")
        print("=" * 60)
        
        print("\n📊 Najbolji rezultati:")
        print(f"   Model sačuvan u: models/trashvision_v1/weights/best.pt")
        
    except Exception as e:
        print(f"\n❌ Greška tokom treniranja: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)

if __name__ == '__main__':
    main()