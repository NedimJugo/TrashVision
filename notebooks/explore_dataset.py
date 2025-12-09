import os
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt

# Putanja do dataseta
dataset_path = Path("data/raw/garbage-dataset")

# Provjeri da li postoji
if not dataset_path.exists():
    print(f"❌ Dataset ne postoji na putanji: {dataset_path}")
    print("Provjeri strukturu foldera:")
    print(os.listdir("data/raw/"))
    exit()

print("✅ Dataset pronađen!")
print(f"📁 Putanja: {dataset_path}\n")

# Analiziraj strukturu
categories = [d for d in dataset_path.iterdir() if d.is_dir()]
print(f"📊 Broj kategorija: {len(categories)}\n")

print("=" * 50)
print("KATEGORIJE I BROJ SLIKA:")
print("=" * 50)

total_images = 0
category_counts = {}

for category in sorted(categories):
    images = list(category.glob("*.jpg")) + list(category.glob("*.png"))
    count = len(images)
    total_images += count
    category_counts[category.name] = count
    print(f"📦 {category.name:20s} - {count:4d} slika")

print("=" * 50)
print(f"📷 UKUPNO SLIKA: {total_images}")
print("=" * 50)

# Vizualizacija
plt.figure(figsize=(12, 6))
plt.bar(category_counts.keys(), category_counts.values(), color='steelblue')
plt.xlabel('Kategorija otpada', fontsize=12)
plt.ylabel('Broj slika', fontsize=12)
plt.title('Distribucija slika po kategorijama', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('data/dataset_distribution.png', dpi=300, bbox_inches='tight')
print("\n✅ Grafikon sačuvan: data/dataset_distribution.png")
plt.show()

# Provjeri dimenzije nekih slika
from PIL import Image
print("\n" + "=" * 50)
print("DIMENZIJE UZORAKA SLIKA:")
print("=" * 50)

for category in sorted(categories)[:3]:  # Prvih 3 kategorije
    images = list(category.glob("*.jpg")) + list(category.glob("*.png"))
    if images:
        img = Image.open(images[0])
        print(f"{category.name:20s} - {img.size[0]}x{img.size[1]} px")