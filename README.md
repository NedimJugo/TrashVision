# 🗑️ TrashVision - AI Waste Classification System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)](https://github.com/ultralytics/ultralytics)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.123.5-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**TrashVision** je napredni AI sistem za automatsku klasifikaciju otpada koji koristi YOLOv8 neuronsku mrežu za prepoznavanje 10 različitih tipova otpada sa detaljnim preporukama za pravilno odlaganje.

![TrashVision Demo](docs/demo.gif)

## ✨ Ključne funkcionalnosti

- 🤖 **AI Klasifikacija** - Prepoznavanje 10 kategorija otpada sa visokom tačnošću (85-92%)
- 🌐 **Web Interfejs** - Moderna, intuitivna aplikacija sa dark/light modom
- ⚡ **REST API** - FastAPI backend za laku integraciju
- 🔄 **Continuous Learning** - Automatsko poboljšanje modela kroz user feedback
- 📊 **Real-time Predikcije** - Brze predikcije (< 1 sekunda) sa confidence skorom
- ♻️ **Preporuke za reciklažu** - Detaljne instrukcije za pravilno odlaganje
- 📱 **Responsive Design** - Perfektno radi na desktop, tablet i mobile uređajima

## 🗂️ Kategorije otpada

| Kategorija | Emoji | Reciklabilno | Odlaganje |
|------------|-------|--------------|-----------|
| Battery | 🔋 | ✅ | Poseban kontejner za baterije |
| Biological | 🌱 | ✅ | Braon/zelena kanta za kompost |
| Cardboard | 📦 | ✅ | Plavi kontejner za papir |
| Clothes | 👕 | ✅ | Donirati ili kontejner za tekstil |
| Glass | 🍾 | ✅ | Zeleni kontejner za staklo |
| Metal | 🔩 | ✅ | Žuti kontejner za metal |
| Paper | 📄 | ✅ | Plavi kontejner za papir |
| Plastic | 🧴 | ✅ | Žuti kontejner za plastiku |
| Shoes | 👟 | ✅ | Donirati ili kontejner za tekstil |
| Trash | 🗑️ | ❌ | Crna/siva kanta za opći otpad |

## 🚀 Brzo pokretanje

### Preduvjeti

- Python 3.10 ili noviji
- pip package manager
- (Opciono) NVIDIA GPU sa CUDA za brže treniranje

### Instalacija

```bash
# 1. Kloniraj repozitorijum
git clone https://github.com/yourusername/trashvision.git
cd trashvision

# 2. Kreiraj virtuelno okruženje
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# 3. Instaliraj zavisnosti
pip install --upgrade pip
pip install -r requirements.txt

# 4. Download pretreniranog YOLOv8 modela
python -c "from ultralytics import YOLO; YOLO('yolov8n-cls.pt')"
```

### Priprema podataka

```bash
# Download Kaggle Garbage Dataset
# https://www.kaggle.com/datasets/mostafaabla/garbage-classification
# Raspakuj u: data/raw/garbage-dataset/

# Pripremi dataset
python src/prepare_data.py
```

### Treniranje modela

```bash
python src/train.py
```

**Napomena**: Treniranje traje ~15-20 minuta na GPU ili ~2-3 sata na CPU.

### Pokretanje aplikacije

```bash
# Pokreni API server
python app/backend/api.py

# Otvori web interfejs u browseru
# app/frontend/index.html
```

API dokumentacija: http://localhost:8000/docs

## 📖 Dokumentacija

Detaljnu dokumentaciju možete pronaći u [DOKUMENTACIJA.md](DOKUMENTACIJA.md) koja uključuje:

- 📐 Arhitekturu sistema
- 🔧 API dokumentaciju
- 🎓 Korak-po-korak tutorial
- 🔄 Continuous Learning setup
- 📊 Performanse i optimizacije
- 🐛 Troubleshooting
- 🚢 Deployment guide

## 📁 Struktura projekta

```
trashvision/
├── app/
│   ├── backend/          # FastAPI server
│   └── frontend/         # Web UI
├── data/
│   ├── processed/        # Train/val/test split
│   ├── raw/             # Originalni dataset
│   └── new_samples/     # Continuous learning samples
├── models/
│   └── trashvision_v1/  # Trenirani model
├── src/
│   ├── train.py         # Model training
│   ├── evaluate.py      # Model evaluacija
│   ├── predict.py       # Standalone predikcija
│   └── continuous_learning.py  # Continuous learning
├── requirements.txt     # Python dependencies
└── README.md           # Ovaj fajl
```

## 🎯 Performanse

| Metrika | Rezultat |
|---------|----------|
| Top-1 Accuracy | 85-92% |
| Top-5 Accuracy | 95-98% |
| Inference (GPU) | 20-50ms |
| Inference (CPU) | 100-200ms |
| Model veličina | ~6.5MB |

### Per-class accuracy

- **Best**: Glass (95%), Cardboard (95%), Metal (90%)
- **Good**: Plastic (90%), Battery (92%), Paper (85%)
- **Challenging**: Clothes (79%), Shoes (80%)

## 🔄 Continuous Learning

TrashVision implementira **incremental learning** - sistem se automatski poboljšava kroz korisničke povratne informacije:

1. **Korisnik uploada sliku** → Model pravi predikciju
2. **Korisnik potvrđuje ili ispravlja** → Sample se čuva
3. **Kada se skupi 100 uzoraka** → Automatski retraining
4. **Model se poboljšava** → Bolja tačnost za budućnost

```bash
# Manuelno pokretanje retraininga
curl -X POST "http://localhost:8000/learning/retrain?epochs=10&mode=incremental"

# Statistika continuous learninga
curl http://localhost:8000/learning/stats
```

## 🛠️ API primjeri

### Predikcija slike

```python
import requests

url = "http://localhost:8000/predict"
files = {'file': open('sample.jpg', 'rb')}
response = requests.post(url, files=files)
result = response.json()

print(f"Klasa: {result['predictions'][0]['name']}")
print(f"Confidence: {result['predictions'][0]['confidence']:.2%}")
```

### cURL primjer

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample.jpg"
```

## 🧪 Testiranje

```bash
# Unit testovi
pytest tests/ -v

# Sa coverage reportom
pytest tests/ --cov=src --cov-report=html

# Load testing
locust -f tests/locustfile.py --host=http://localhost:8000
```

## 📦 Deployment

### Docker

```bash
# Build Docker image
docker build -t trashvision:latest .

# Run container
docker run -p 8000:8000 trashvision:latest
```

### Cloud platforms

- **AWS**: EC2 + S3
- **Google Cloud**: Cloud Run
- **Heroku**: Git deployment
- **Azure**: App Service

Detaljne deployment instrukcije u [DOKUMENTACIJA.md](DOKUMENTACIJA.md#-deployment).

## 🛠️ Troubleshooting

### Model nije pronađen
```bash
# Pokreni treniranje prvo
python src/train.py
```

### CUDA out of memory
```python
# Smanji batch size u src/train.py
BATCH_SIZE = 16  # ili 8
```

### API ne reaguje
```bash
# Provjeri port 8000
netstat -ano | findstr :8000
```

Više rješenja u [DOKUMENTACIJA.md](DOKUMENTACIJA.md#-troubleshooting).

## 🤝 Doprinos

Dobrodošli su svi doprinosi! Ako želite doprinijeti:

1. Fork-ujte projekat
2. Kreirajte feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit-ujte promjene (`git commit -m 'Add AmazingFeature'`)
4. Push-ujte na branch (`git push origin feature/AmazingFeature`)
5. Otvorite Pull Request

### Development guidelines

- Slijedite PEP 8 style guide
- Dodajte testove za nove features
- Update-ujte dokumentaciju
- Pišite opisne commit poruke

## 📚 Resursi

### Dokumentacija
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)

### Datasets
- [Kaggle Garbage Dataset](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)
- [TACO Dataset](http://tacodataset.org/)
- [TrashNet](https://github.com/garythung/trashnet)

### Research Papers
- [YOLOv8](https://github.com/ultralytics/ultralytics)
- [Lifelong Machine Learning](https://www.cs.uic.edu/~liub/lifelong-learning.html)
- [Deep Learning for Waste Classification](https://arxiv.org/abs/2007.08303)

## 📄 Licenca

Ovaj projekat je licenciran pod MIT licencom - pogledajte [LICENSE](LICENSE) fajl za detalje.

**Napomena**: YOLOv8 je pod AGPL-3.0 licencom za nekomercijalnu upotrebu.

## 🙏 Zahvalnice

- **Ultralytics** za YOLOv8 framework
- **Kaggle** za garbage classification dataset
- **FastAPI** za odličan web framework
- **Community** za feedback i doprinose

## 📞 Kontakt

- **GitHub Issues**: [github.com/NedimJugo/TrashVision/issues](https://github.com/NedimJugo/TrashVision/issues)
- **Email**: nedim.jugoo@gmail.com

---

<div align="center">

**TrashVision** - Čineći svijet čistijim, jednu sliku po sliku. 🌍♻️

</div>
