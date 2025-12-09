from ultralytics import YOLO
import torch
from pathlib import Path
import shutil
from datetime import datetime
import json
from sklearn.model_selection import train_test_split

class ContinuousLearner:
    def __init__(self, base_model_path="models/trashvision_v1/weights/best.pt"):
        self.base_model_path = Path(base_model_path)
        self.new_data_dir = Path("data/new_samples")
        self.feedback_file = Path("data/user_feedback.json")
        self.config_file = Path("data/learning_config.json")
        
        self.new_data_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.config = self._load_config()
    
    def _load_config(self):
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return json.load(f)
        else:
            default_config = {
                "auto_retrain_threshold": 100,
                "current_samples": 0,
                "last_retrain": None,
                "retrain_count": 0,
                "min_confidence_for_auto_add": 0.85,
                "retrain_mode": "incremental"  # 'full' ili 'incremental'
            }
            self._save_config(default_config)
            return default_config
    
    def _save_config(self, config=None):
        if config:
            self.config = config
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def add_sample(self, image_data, predicted_class, confidence, user_confirmed_class=None):
        from PIL import Image
        import io
        
        final_class = user_confirmed_class if user_confirmed_class else predicted_class
        
        class_dir = self.new_data_dir / final_class
        class_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{final_class}_{timestamp}.jpg"
        filepath = class_dir / filename
        
        if isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data))
        else:
            image = image_data
        
        image.save(filepath, 'JPEG', quality=95)
        
        self._log_feedback(filepath, predicted_class, final_class, confidence)
        
        self.config["current_samples"] += 1
        self._save_config()
        
        print(f"✅ Dodata nova slika: {filename}")
        print(f"📊 Ukupno novih uzoraka: {self.config['current_samples']}")
        
        if self.config["current_samples"] >= self.config["auto_retrain_threshold"]:
            return True
        
        return False
    
    def _log_feedback(self, filepath, predicted, actual, confidence):
        feedback = []
        if self.feedback_file.exists():
            with open(self.feedback_file, 'r') as f:
                feedback = json.load(f)
        
        feedback.append({
            "timestamp": datetime.now().isoformat(),
            "filepath": str(filepath),
            "predicted_class": predicted,
            "actual_class": actual,
            "confidence": float(confidence),
            "was_correct": predicted == actual
        })
        
        with open(self.feedback_file, 'w') as f:
            json.dump(feedback, f, indent=2)
    
    def prepare_incremental_dataset(self):
        """
        ✅ NOVA VERZIJA: Pripremi SAMO nove podatke za incremental training
        """
        temp_dir = Path("data/incremental_train")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        categories = [d.name for d in self.new_data_dir.iterdir() if d.is_dir()]
        
        print(f"📦 Pripremam incremental dataset: {len(categories)} kategorija")
        
        for category in categories:
            new_images = list((self.new_data_dir / category).glob("*.jpg"))
            
            if not new_images:
                continue
            
            print(f"  {category}: {len(new_images)} novih slika")
            
            # Split samo novih podataka
            train_imgs, val_imgs = train_test_split(
                new_images, 
                test_size=0.2, 
                random_state=42
            )
            
            # Kopiraj u temp folder
            for split, images in [("train", train_imgs), ("val", val_imgs)]:
                target_dir = temp_dir / split / category
                target_dir.mkdir(parents=True, exist_ok=True)
                
                for img in images:
                    shutil.copy(img, target_dir / img.name)
        
        print("✅ Incremental dataset spreman")
        return temp_dir
    
    def retrain_incremental(self, epochs=10, backup_old=True):
        """
        ✅ BOLJA VERZIJA: Incremental/Fine-tuning samo na novim podacima
        
        Prednosti:
        - 10x brže nego full retrain
        - Manje memorije
        - Smanjuje catastrophic forgetting
        """
        print("=" * 60)
        print("🔄 INCREMENTAL RETRAINING...")
        print("=" * 60)
        
        if backup_old:
            backup_dir = Path("models/backups")
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"model_backup_{timestamp}.pt"
            shutil.copy(self.base_model_path, backup_path)
            print(f"💾 Backup: {backup_path}")
        
        # Pripremi SAMO nove podatke
        temp_dataset = self.prepare_incremental_dataset()
        
        # Load trenutni model
        model = YOLO(str(self.base_model_path))
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"\n🔥 Fine-tuning sa {epochs} epoha (SAMO novi podaci)...")
        
        # ✅ KLJUČNO: Niži learning rate + freeze backbone (opciono)
        results = model.train(
            data=str(temp_dataset),  # ← SAMO novi podaci
            epochs=epochs,
            batch=16,  # Manji batch za stabilnost
            imgsz=224,
            device=device,
            patience=3,
            save=True,
            project='models',
            name=f'trashvision_incremental_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            exist_ok=True,
            pretrained=True,
            optimizer='Adam',
            verbose=True,
            workers=0,
            
            # ✅ Fine-tuning hiperparametri
            lr0=0.00001,  # VRLO nizak LR da ne "zaboravi" staro znanje
            lrf=0.0001,
            weight_decay=0.001,
            warmup_epochs=1,
            
            # ✅ Data augmentation za bolje generalizovanje
            augment=True,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=15,
            translate=0.1,
            scale=0.5,
            fliplr=0.5,
        )
        
        # Očisti temp folder
        shutil.rmtree(temp_dataset)
        
        self.config["current_samples"] = 0
        self.config["last_retrain"] = datetime.now().isoformat()
        self.config["retrain_count"] += 1
        self._save_config()
        
        # Arhiviraj new_samples
        archive_dir = Path("data/archived_samples") / datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.move(str(self.new_data_dir), str(archive_dir))
        self.new_data_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "=" * 60)
        print("✅ INCREMENTAL RETRAINING ZAVRŠEN!")
        print(f"📊 Retraining broj: {self.config['retrain_count']}")
        print("=" * 60)
        
        return results
    
    def retrain_full(self, epochs=20, backup_old=True):
        """
        ⚠️ STARA VERZIJA: Full retrain (SPORO, ali najpreciznije)
        
        Koristi samo kad:
        - Imaš puno novih podataka (500+)
        - Model pokazuje značajan pad tačnosti
        - Imaš dovoljno GPU resursa
        """
        print("=" * 60)
        print("🔄 FULL RETRAINING (može trajati dugo)...")
        print("=" * 60)
        
        if backup_old:
            backup_dir = Path("models/backups")
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"model_backup_{timestamp}.pt"
            shutil.copy(self.base_model_path, backup_path)
            print(f"💾 Backup: {backup_path}")
        
        # Integriši SVE podatke
        self.prepare_full_dataset()
        
        model = YOLO(str(self.base_model_path))
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"\n🔥 Full retraining sa {epochs} epoha...")
        
        results = model.train(
            data='data/processed',  # ← CIJELI dataset
            epochs=epochs,
            batch=32,
            imgsz=224,
            device=device,
            patience=5,
            save=True,
            project='models',
            name=f'trashvision_full_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            exist_ok=True,
            pretrained=True,
            optimizer='Adam',
            verbose=True,
            workers=0,
        )
        
        self.config["current_samples"] = 0
        self.config["last_retrain"] = datetime.now().isoformat()
        self.config["retrain_count"] += 1
        self._save_config()
        
        archive_dir = Path("data/archived_samples") / datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.move(str(self.new_data_dir), str(archive_dir))
        self.new_data_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "=" * 60)
        print("✅ FULL RETRAINING ZAVRŠEN!")
        print("=" * 60)
        
        return results
    
    def prepare_full_dataset(self):
        """Helper za full retrain - integriše nove u processed"""
        processed_dir = Path("data/processed")
        
        categories = [d.name for d in self.new_data_dir.iterdir() if d.is_dir()]
        
        print(f"📦 Integriše {len(categories)} kategorija u processed dataset")
        
        for category in categories:
            new_images = list((self.new_data_dir / category).glob("*.jpg"))
            
            if not new_images:
                continue
            
            train_imgs, temp = train_test_split(new_images, test_size=0.3, random_state=42)
            val_imgs, test_imgs = train_test_split(temp, test_size=0.5, random_state=42)
            
            for split, images in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
                target_dir = processed_dir / split / category
                target_dir.mkdir(parents=True, exist_ok=True)
                
                for img in images:
                    shutil.copy(img, target_dir / img.name)
        
        print("✅ Podaci integrisani")
    
    def retrain_model(self, epochs=None, mode=None):
        """
        ✅ GLAVNA METODA: Automatski bira najbolji pristup
        """
        mode = mode or self.config.get("retrain_mode", "incremental")
        
        if mode == "incremental":
            epochs = epochs or 10
            return self.retrain_incremental(epochs=epochs)
        else:
            epochs = epochs or 20
            return self.retrain_full(epochs=epochs)
    
    def get_stats(self):
        feedback_data = []
        if self.feedback_file.exists():
            with open(self.feedback_file, 'r') as f:
                feedback_data = json.load(f)
        
        correct = sum(1 for f in feedback_data if f['was_correct'])
        total = len(feedback_data)
        accuracy = (correct / total * 100) if total > 0 else 0
        
        return {
            "new_samples": self.config["current_samples"],
            "auto_retrain_threshold": self.config["auto_retrain_threshold"],
            "total_feedback": total,
            "accuracy": accuracy,
            "last_retrain": self.config.get("last_retrain"),
            "retrain_count": self.config["retrain_count"],
            "retrain_mode": self.config.get("retrain_mode", "incremental")
        }