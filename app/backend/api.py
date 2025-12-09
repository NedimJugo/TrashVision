import sys
from pathlib import Path

# Dodaj root direktorij u sys.path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
from src.continuous_learning import ContinuousLearner
import io
import numpy as np
from pathlib import Path
import base64
import traceback

learner = ContinuousLearner()

# Inicijalizuj FastAPI
app = FastAPI(title="TrashVision API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Učitaj model pri startu aplikacije
MODEL_PATH = Path("models/trashvision_v1/weights/best.pt")

# Provjeri da li model postoji
if not MODEL_PATH.exists():
    print(f"❌ GREŠKA: Model ne postoji na putanji: {MODEL_PATH}")
    print(f"📁 Trenutni direktorijum: {Path.cwd()}")
    print(f"📂 Sadržaj models foldera:")
    if Path("models").exists():
        for item in Path("models").rglob("*.pt"):
            print(f"   - {item}")
    raise FileNotFoundError(f"Model nije pronađen: {MODEL_PATH}")

print(f"✅ Učitavam model: {MODEL_PATH}")
model = YOLO(str(MODEL_PATH))
print(f"✅ Model uspješno učitan!")

# Klase i preporuke
CLASSES_INFO = {
    'battery': {
        'name': 'Battery (Baterija)',
        'disposal': 'Poseban kontejner za baterije ili reciklažno dvorište',
        'recyclable': True,
        'color': '#ff9800'
    },
    'biological': {
        'name': 'Biological (Organski otpad)',
        'disposal': 'Braon/zelena kanta za kompost',
        'recyclable': True,
        'color': '#4caf50'
    },
    'cardboard': {
        'name': 'Cardboard (Karton)',
        'disposal': 'Plavi kontejner za papir i karton',
        'recyclable': True,
        'color': '#2196f3'
    },
    'clothes': {
        'name': 'Clothes (Odjeća)',
        'disposal': 'Donirati ili odvesti u kontejner za tekstil',
        'recyclable': True,
        'color': '#e91e63'
    },
    'glass': {
        'name': 'Glass (Staklo)',
        'disposal': 'Zeleni kontejner za staklo',
        'recyclable': True,
        'color': '#00bcd4'
    },
    'metal': {
        'name': 'Metal',
        'disposal': 'Žuti kontejner za metal',
        'recyclable': True,
        'color': '#9e9e9e'
    },
    'paper': {
        'name': 'Paper (Papir)',
        'disposal': 'Plavi kontejner za papir',
        'recyclable': True,
        'color': '#2196f3'
    },
    'plastic': {
        'name': 'Plastic (Plastika)',
        'disposal': 'Žuti kontejner za plastiku',
        'recyclable': True,
        'color': '#ffeb3b'
    },
    'shoes': {
        'name': 'Shoes (Obuća)',
        'disposal': 'Donirati ili odvesti u kontejner za tekstil',
        'recyclable': True,
        'color': '#795548'
    },
    'trash': {
        'name': 'Trash (Mješoviti otpad)',
        'disposal': 'Crna/siva kanta za opći otpad',
        'recyclable': False,
        'color': '#607d8b'
    }
}

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "TrashVision API is running",
        "model": "YOLOv8n-cls",
        "version": "1.0.0"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Klasifikuj sliku otpada"""
    try:
        print(f"📥 Primljena slika: {file.filename}, tip: {file.content_type}")
        
        # Provjeri format fajla
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Fajl mora biti slika")
        
        # Učitaj sliku
        image_data = await file.read()
        print(f"📊 Veličina slike: {len(image_data)} bytes")
        
        image = Image.open(io.BytesIO(image_data))
        print(f"🖼️  Dimenzije: {image.size}, Mode: {image.mode}")
        
        # Konvertuj u RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
            print(f"✅ Konvertovano u RGB")
        
        # Predikcija
        print("🔮 Pokrećem predikciju...")
        results = model(image, verbose=False)[0]
        probs = results.probs
        print(f"✅ Predikcija završena")
        
        # Top 3 predikcije
        top3_indices = probs.top5[:3]
        top3_probs = probs.data[top3_indices]
        
        # Konvertuj u numpy ako je tensor
        if hasattr(top3_indices, 'cpu'):
            top3_indices = top3_indices.cpu().numpy()
        if hasattr(top3_probs, 'cpu'):
            top3_probs = top3_probs.cpu().numpy()
        
        classes = list(CLASSES_INFO.keys())
        
        # Pripremi rezultat
        predictions = []
        for idx, prob in zip(top3_indices, top3_probs):
            class_key = classes[int(idx)]
            class_info = CLASSES_INFO[class_key]
            predictions.append({
                'class': class_key,
                'name': class_info['name'],
                'confidence': float(prob),
                'disposal': class_info['disposal'],
                'recyclable': class_info['recyclable'],
                'color': class_info['color']
            })
        
        print(f"📊 Top predikcija: {predictions[0]['name']} ({predictions[0]['confidence']*100:.1f}%)")
        
        # Konvertuj sliku u base64
        buffered = io.BytesIO()
        image.thumbnail((400, 400))
        image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        return JSONResponse({
            'success': True,
            'predictions': predictions,
            'image': f"data:image/jpeg;base64,{img_base64}"
        })
    
    except Exception as e:
        print(f"❌ GREŠKA: {str(e)}")
        print(f"📋 Traceback:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Greška: {str(e)}")

@app.post("/feedback")
async def submit_feedback(
    file: UploadFile = File(...),
    predicted_class: str = Form(...),
    actual_class: str = Form(...),
    confidence: float = Form(...)
):
    """Korisnik potvrđuje ili ispravlja predikciju"""
    try:
        image_data = await file.read()
        
        # Dodaj u dataset
        should_retrain = learner.add_sample(
            image_data, 
            predicted_class, 
            confidence, 
            user_confirmed_class=actual_class
        )
        
        return JSONResponse({
            'success': True,
            'message': 'Hvala na feedbacku!',
            'should_retrain': should_retrain,
            'new_samples_count': learner.config["current_samples"]
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/learning/stats")
async def get_learning_stats():
    """Vrati statistiku continuous learninga"""
    stats = learner.get_stats()
    return stats


@app.post("/learning/retrain")
async def trigger_retrain(epochs: int = 10, mode: str = "incremental"):
    """Manuelno pokreni retraining"""
    try:
        # mode = "incremental" ili "full"
        results = learner.retrain_model(epochs=epochs, mode=mode)
        return JSONResponse({
            'success': True,
            'message': f'Retraining završen! ({mode} mode)',
            'retrain_count': learner.config["retrain_count"]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/learning/config")
async def get_config():
    """Vrati trenutnu konfiguraciju"""
    return learner.config


@app.post("/learning/config")
async def update_config(config: dict):
    """Update konfiguraciju"""
    learner._save_config(config)
    return {'success': True, 'config': learner.config}

@app.get("/classes")
async def get_classes():
    """Vrati sve klase"""
    return CLASSES_INFO

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("🚀 Pokrećem TrashVision API...")
    print("=" * 60)
    print(f"📍 URL: http://localhost:8000")
    print(f"📚 Docs: http://localhost:8000/docs")
    print(f"📁 Radni direktorijum: {Path.cwd()}")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)