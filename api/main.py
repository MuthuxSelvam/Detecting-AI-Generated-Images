from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import sys

# Ensure root directory is in python path to import ml
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.model import EnsembleDetector
from ml.analysis import fft_analysis, noise_residual_analysis

MODEL_PATH = "best_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

# Transform for inference (same as validation)
inference_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

def load_model_weights():
    """Load model weights at startup"""
    global model
    try:
        # Initialize model
        model = EnsembleDetector(pretrained=False) 
        
        if os.path.exists(MODEL_PATH):
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
            print(f"Loaded model from {MODEL_PATH}")
        else:
            print(f"Warning: {MODEL_PATH} not found. Using random weights.")
        
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern lifespan context manager for startup/shutdown"""
    load_model_weights()
    yield

app = FastAPI(title="AI Image Detector", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "AI Image Detector API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Read Image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        # 1. Analysis streams (FFT/Noise)
        img_resized = cv2.resize(image_rgb, (224, 224))
        
        fft_map = fft_analysis(img_resized)
        fft_tensor = torch.tensor(fft_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) # (1, 1, H, W)
        
        noise_map = noise_residual_analysis(img_resized)
        noise_tensor = torch.tensor(noise_map, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) # (1, 3, H, W)
        
        # 2. RGB stream
        augmented = inference_transform(image=image_rgb)['image']
        rgb_tensor = augmented.unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            output = model(rgb_tensor, fft_tensor, noise_tensor)
            prob = torch.sigmoid(output).item()
            
        return {
            "filename": file.filename,
            "is_fake": prob > 0.5,
            "probability": prob,
            "label": "Fake" if prob > 0.5 else "Real"
        }
    except Exception as e:
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
