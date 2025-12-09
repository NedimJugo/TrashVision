from ultralytics import YOLO
import torch
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from PIL import Image

def main():
    print("=" * 60)
    print("📊 TrashVision - Model Evaluation")
    print("=" * 60)
    
    # Učitaj najbolji model
    model_path = "models/trashvision_v1/weights/best.pt"
    print(f"\n📦 Učitavam model: {model_path}")
    model = YOLO(model_path)
    
    # Test dataset
    test_dir = Path("data/processed/test")
    
    # Klase
    classes = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])
    print(f"\n🏷️  Klase: {classes}")
    
    # Validiraj na test setu
    print("\n🔍 Evaluacija na test setu...\n")
    results = model.val(data="data/processed", split="test")
    
    print("\n" + "=" * 60)
    print("📈 TEST SET REZULTATI:")
    print("=" * 60)
    print(f"Top-1 Accuracy: {results.top1:.3f} ({results.top1*100:.1f}%)")
    print(f"Top-5 Accuracy: {results.top5:.3f} ({results.top5*100:.1f}%)")
    print("=" * 60)
    
    # Detaljnija analiza - predikcija po klasama
    print("\n🔬 Detaljnja analiza po klasama...\n")
    
    y_true = []
    y_pred = []
    
    for class_idx, class_name in enumerate(classes):
        class_path = test_dir / class_name
        images = list(class_path.glob("*.jpg")) + list(class_path.glob("*.png"))
        
        for img_path in images:
            # Predikcija
            result = model(img_path, verbose=False)[0]
            predicted_class = result.probs.top1
            
            y_true.append(class_idx)
            y_pred.append(predicted_class)
    
    # Classification report
    print("\n" + "=" * 60)
    print("📊 CLASSIFICATION REPORT:")
    print("=" * 60)
    report = classification_report(y_true, y_pred, target_names=classes, digits=3)
    print(report)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plotuj confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Broj slika'})
    plt.title('Confusion Matrix - Test Set', fontsize=16, fontweight='bold')
    plt.ylabel('Stvarna klasa', fontsize=12)
    plt.xlabel('Predviđena klasa', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Sačuvaj
    output_path = "models/trashvision_v1/confusion_matrix.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Confusion matrix sačuvana: {output_path}")
    plt.show()
    
    # Per-class accuracy
    print("\n" + "=" * 60)
    print("📊 TAČNOST PO KLASAMA:")
    print("=" * 60)
    
    class_correct = cm.diagonal()
    class_total = cm.sum(axis=1)
    class_accuracy = class_correct / class_total
    
    for i, class_name in enumerate(classes):
        acc = class_accuracy[i] * 100
        print(f"{class_name:15s}: {acc:5.1f}% ({class_correct[i]:3d}/{class_total[i]:3d})")
    
    print("=" * 60)
    
    # Pronađi najgore klasifikovane primjere
    print("\n🔍 Najčešće greške:")
    errors = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i, j] > 0:
                errors.append((classes[i], classes[j], cm[i, j]))
    
    errors.sort(key=lambda x: x[2], reverse=True)
    
    print("\nTop 5 konfuzija:")
    for true_class, pred_class, count in errors[:5]:
        print(f"  {true_class:12s} → {pred_class:12s}: {count:3d} puta")
    
    print("\n" + "=" * 60)
    print("✅ Evaluacija završena!")
    print("=" * 60)

if __name__ == '__main__':
    main()