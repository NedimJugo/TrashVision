import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import random

# Postavi seed za reproducibilnost
random.seed(42)

# Putanje
RAW_DATA = Path("data/raw/garbage-dataset")
PROCESSED_DATA = Path("data/processed")

# Kreiraj strukturu foldera
TRAIN_DIR = PROCESSED_DATA / "train"
VAL_DIR = PROCESSED_DATA / "val"
TEST_DIR = PROCESSED_DATA / "test"

for dir_path in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

print("🚀 Započinjem pripremu podataka...\n")

# Kategorije
categories = [d.name for d in RAW_DATA.iterdir() if d.is_dir()]
print(f"📦 Pronađeno {len(categories)} kategorija: {categories}\n")

# Statistika
stats = {
    'train': {},
    'val': {},
    'test': {}
}

for category in sorted(categories):
    print(f"⚙️  Obrađujem kategoriju: {category}")
    
    # Učitaj sve slike iz kategorije
    category_path = RAW_DATA / category
    images = list(category_path.glob("*.jpg")) + list(category_path.glob("*.png"))
    
    print(f"   📷 Ukupno slika: {len(images)}")
    
    # Podijeli podatke: 70% train, 15% val, 15% test
    train_images, temp_images = train_test_split(images, test_size=0.3, random_state=42)
    val_images, test_images = train_test_split(temp_images, test_size=0.5, random_state=42)
    
    # Kreiraj foldere za kategorije
    (TRAIN_DIR / category).mkdir(exist_ok=True)
    (VAL_DIR / category).mkdir(exist_ok=True)
    (TEST_DIR / category).mkdir(exist_ok=True)
    
    # Kopiraj slike
    for img in train_images:
        shutil.copy(img, TRAIN_DIR / category / img.name)
    
    for img in val_images:
        shutil.copy(img, VAL_DIR / category / img.name)
    
    for img in test_images:
        shutil.copy(img, TEST_DIR / category / img.name)
    
    # Sačuvaj statistiku
    stats['train'][category] = len(train_images)
    stats['val'][category] = len(val_images)
    stats['test'][category] = len(test_images)
    
    print(f"   ✅ Train: {len(train_images)} | Val: {len(val_images)} | Test: {len(test_images)}\n")

# Ispiši sveukupnu statistiku
print("=" * 60)
print("📊 FINALNA STATISTIKA:")
print("=" * 60)

for split in ['train', 'val', 'test']:
    total = sum(stats[split].values())
    print(f"\n{split.upper()}:")
    for category, count in sorted(stats[split].items()):
        print(f"  {category:15s}: {count:4d} slika")
    print(f"  {'UKUPNO':15s}: {total:4d} slika")

print("\n" + "=" * 60)
print("✅ Priprema podataka završena!")
print("=" * 60)

# Sačuvaj label mappings
label_map = {i: cat for i, cat in enumerate(sorted(categories))}
print("\n📝 Label mapping:")
for idx, name in label_map.items():
    print(f"  {idx}: {name}")

# Sačuvaj u fajl
with open(PROCESSED_DATA / "labels.txt", "w") as f:
    for idx, name in label_map.items():
        f.write(f"{idx}:{name}\n")

print(f"\n✅ Label mapping sačuvan: {PROCESSED_DATA / 'labels.txt'}")